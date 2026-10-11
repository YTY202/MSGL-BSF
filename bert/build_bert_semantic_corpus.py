import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


HMDB_NS = "{http://www.hmdb.ca}"
MESH_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def child_text(element, path, namespace=""):
    child = element.find(path if not namespace else namespace + path)
    return clean_text(child.text) if child is not None else ""


def child_texts(element, path):
    return [clean_text(item.text) for item in element.findall(path) if clean_text(item.text)]


def normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())


def extract_hmdb(zip_path, target_ids, target_names):
    records = {}
    with zipfile.ZipFile(zip_path) as archive:
        xml_name = next(name for name in archive.namelist() if name.lower().endswith(".xml"))
        with archive.open(xml_name) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != HMDB_NS + "metabolite":
                    continue
                accession = child_text(element, "accession", HMDB_NS)
                secondary = child_texts(element, f"{HMDB_NS}secondary_accessions/{HMDB_NS}accession")
                matched_ids = ({accession, *secondary} & target_ids)
                hmdb_name = child_text(element, "name", HMDB_NS)
                name_matched_ids = target_names.get(normalized_name(hmdb_name), set())
                matched_ids.update(name_matched_ids - records.keys())
                if matched_ids:
                    taxonomy = element.find(HMDB_NS + "taxonomy")
                    taxonomy_values = []
                    if taxonomy is not None:
                        for field in ("kingdom", "super_class", "class", "sub_class", "direct_parent", "molecular_framework"):
                            value = child_text(taxonomy, field, HMDB_NS)
                            if value:
                                taxonomy_values.append(value)
                    synonyms = child_texts(element, f"{HMDB_NS}synonyms/{HMDB_NS}synonym")[:12]
                    locations = []
                    for group in ("cellular_locations", "biospecimen_locations", "tissue_locations"):
                        locations.extend(child_texts(element, f"{HMDB_NS}biological_properties/{HMDB_NS}{group}/{HMDB_NS}{group[:-1]}"))
                    record = {
                        "HMDB Name": hmdb_name,
                        "Description": child_text(element, "description", HMDB_NS),
                        "Synonyms": "; ".join(dict.fromkeys(synonyms)),
                        "Chemical Formula": child_text(element, "chemical_formula", HMDB_NS),
                        "IUPAC Name": child_text(element, "iupac_name", HMDB_NS),
                        "SMILES": child_text(element, "smiles", HMDB_NS),
                        "Taxonomy": "; ".join(dict.fromkeys(taxonomy_values)),
                        "Biological Locations": "; ".join(dict.fromkeys(locations)),
                    }
                    for matched_id in matched_ids:
                        records[matched_id] = {
                            **record,
                            "Match Method": "HMDB ID" if matched_id in {accession, *secondary} else "Exact Name",
                        }
                element.clear()
                if len(records) == len(target_ids):
                    break
    return records


def download_mesh(mesh_path):
    if mesh_path.exists():
        return
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mesh_path.with_suffix(mesh_path.suffix + ".part")
    urllib.request.urlretrieve(MESH_URL, temporary)
    temporary.replace(mesh_path)


def extract_mesh(mesh_path, target_ids):
    records = {}
    for _, element in ET.iterparse(mesh_path, events=("end",)):
        if element.tag != "DescriptorRecord":
            continue
        descriptor_id = child_text(element, "DescriptorUI")
        if descriptor_id in target_ids:
            preferred_name = child_text(element, "DescriptorName/String")
            concept = element.find("ConceptList/Concept[@PreferredConceptYN='Y']")
            if concept is None:
                concept = element.find("ConceptList/Concept")
            scope_note = child_text(concept, "ScopeNote") if concept is not None else ""
            terms = child_texts(element, "ConceptList/Concept/TermList/Term/String")
            tree_numbers = child_texts(element, "TreeNumberList/TreeNumber")
            records[descriptor_id] = {
                "MeSH Preferred Name": preferred_name,
                "Scope Note": scope_note,
                "Synonyms": "; ".join(dict.fromkeys(terms)),
                "Tree Numbers": "; ".join(dict.fromkeys(tree_numbers)),
            }
        element.clear()
    return records


def join_nonempty(parts):
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def main():
    parser = argparse.ArgumentParser(description="Build aligned HMDB and MeSH text corpora for BERT encoding.")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--hmdb_zip", required=True)
    parser.add_argument("--output_dir", default="data1")
    parser.add_argument("--mesh_xml", default="result/source_cache/desc2026.xml")
    args = parser.parse_args()

    workbook = Path(args.workbook)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = pd.ExcelFile(workbook).sheet_names
    disease_list = pd.read_excel(workbook, sheet_name=2, dtype=str).fillna("")
    metabolite_list = pd.read_excel(workbook, sheet_name=3, dtype=str).fillna("")

    metabolite_ids = set(metabolite_list["Metabolite ID"].map(clean_text))
    metabolite_names = {}
    for _, row in metabolite_list.iterrows():
        metabolite_names.setdefault(normalized_name(row["Metabolite Name"]), set()).add(clean_text(row["Metabolite ID"]))
    mesh_ids = set(disease_list["Mesh ID"].map(clean_text))
    hmdb_records = extract_hmdb(Path(args.hmdb_zip), metabolite_ids, metabolite_names)

    mesh_path = Path(args.mesh_xml)
    download_mesh(mesh_path)
    mesh_records = extract_mesh(mesh_path, mesh_ids)

    metabolite_rows = []
    for position, row in metabolite_list.reset_index(drop=True).iterrows():
        metabolite_id = clean_text(row["Metabolite ID"])
        source = hmdb_records.get(metabolite_id, {})
        dataset_name = clean_text(row["Metabolite Name"])
        dataset_smiles = clean_text(row.get("SMILES", ""))
        bert_text = join_nonempty([
            f"Metabolite: {dataset_name}.",
            f"HMDB preferred name: {source.get('HMDB Name', '')}.",
            f"Chemical class: {source.get('Taxonomy', '')}.",
            f"IUPAC name: {source.get('IUPAC Name', '')}.",
            f"Formula: {source.get('Chemical Formula', '')}.",
            f"SMILES: {source.get('SMILES', '') or dataset_smiles}.",
            f"Biological locations: {source.get('Biological Locations', '')}.",
            f"Synonyms: {source.get('Synonyms', '')}.",
            source.get("Description", ""),
        ])
        metabolite_rows.append({
            "Matrix Row": position,
            "Metabolite Name": dataset_name,
            "Metabolite ID": metabolite_id,
            "Dataset SMILES": dataset_smiles,
            **source,
            "BERT Text": bert_text,
            "Source URL": f"https://hmdb.ca/metabolites/{metabolite_id}",
            "Matched": bool(source),
        })

    disease_rows = []
    for position, row in disease_list.reset_index(drop=True).iterrows():
        mesh_id = clean_text(row["Mesh ID"])
        source = mesh_records.get(mesh_id, {})
        dataset_name = clean_text(row["Disease Name"])
        bert_text = join_nonempty([
            f"Disease: {dataset_name}.",
            f"MeSH preferred name: {source.get('MeSH Preferred Name', '')}.",
            source.get("Scope Note", ""),
            f"MeSH terms: {source.get('Synonyms', '')}.",
        ])
        disease_rows.append({
            "Matrix Column": position,
            "Disease Name": dataset_name,
            "Mesh ID": mesh_id,
            **source,
            "BERT Text": bert_text,
            "Source URL": f"https://id.nlm.nih.gov/mesh/{mesh_id}.html",
            "Matched": bool(source),
        })

    metabolite_output = pd.DataFrame(metabolite_rows)
    disease_output = pd.DataFrame(disease_rows)
    metabolite_output.to_csv(output_dir / "metabolite_bert_corpus.csv", index=False, encoding="utf-8-sig")
    disease_output.to_csv(output_dir / "disease_bert_corpus.csv", index=False, encoding="utf-8-sig")
    summary = {
        "workbook": str(workbook),
        "workbook_sheets": sheets,
        "hmdb_source": str(Path(args.hmdb_zip)),
        "mesh_source": MESH_URL,
        "metabolites": len(metabolite_output),
        "metabolites_matched": int(metabolite_output["Matched"].sum()),
        "diseases": len(disease_output),
        "unique_mesh_ids": len(mesh_ids),
        "diseases_matched": int(disease_output["Matched"].sum()),
    }
    (output_dir / "bert_corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
