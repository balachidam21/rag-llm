from pathlib import Path
from rag.config import EFS_DIR
from bs4 import BeautifulSoup, NavigableString
import bs4

def extract_text_from_section(section:bs4.element.Tag):
    texts = []
    for element in section.children:
        if isinstance(element, NavigableString):
            if element.strip():
                texts.append(element.strip())
        elif element.name == "section":
            continue
        else:
            texts.append(element.get_text().strip())
    return "\n".join(texts)

def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]

def extract_sections(record: dict):
    with open(record["path"], "r", encoding="utf-8") as fp:
        soup = BeautifulSoup(fp, "html.parser")

    sections = soup.find_all("section")
    section_list = []
    for section in sections:
        section_id = section.get("id")
        section_text = extract_text_from_section(section)
        if section_id:
            uri = path_to_uri(path=record["path"])
            section_list.append({"source": f"{uri}#{section_id}", "text": section_text})
    return section_list
