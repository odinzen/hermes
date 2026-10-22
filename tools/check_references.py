"""Check every reference in the README and docs/validation.md against its registry record (Crossref, DataCite, arXiv).

Each entry holds the details as the README cites them. The script fetches the record and compares the full author
list in order, the title, year, volume and first page; a live DOI alone is not taken as a check. Needs network
access; run it before a release.

    python tools/check_references.py
"""

import json
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

# (identifier, authors' family names in order, title, year, volume, first page) as the README cites them
REFERENCES = [
    ("arXiv:2512.02221", ["Fisher", "Hong"],
     "Thermodynamic Entropy as Information -- A compression-based demonstration of the Shannon-Boltzmann "
     "equivalence in condensed matter", 2025, None, None),
    ("10.5281/zenodo.18262305", ["Hew"], "pyzentropy", None, None, None),
    ("10.5281/zenodo.23423225", ["Bustamante", "Hew"], "Hermes: an all-in-one entropy tool", None, None, None),
    ("10.1063/1.1744724", ["Nettleton", "Green"],
     "Expression in Terms of Molecular Distribution Functions for the Entropy Density in an Infinite System",
     1958, "29", "1365"),
    ("10.1103/PhysRevA.40.3817", ["Baranyai", "Evans"], "Direct entropy calculation from computer simulation of liquids",
     1989, "40", "3817"),
    ("10.1103/PhysRevB.2.3004", ["Bhatia", "Thornton"], "Structural Aspects of the Electrical Resistivity of Binary Alloys",
     1970, "2", "3004"),
    ("10.1557/mrc.2019.59", ["Bocklund", "Otis", "Egorov", "Obaied", "Roslyakova", "Liu"],
     "ESPEI for efficient thermodynamic database development, modification, and uncertainty quantification: "
     "application to Cu-Mg", 2019, "9", "618"),
    ("10.1103/PhysRevMaterials.5.103801", ["Menon", "Lysogorskiy", "Rogal", "Drautz"],
     "Automated free-energy calculation from atomistic simulations", 2021, "5", "103801"),
    ("10.1080/14786430902832773", ["Mendelev", "Kramer", "Ott", "Sordelet", "Yagodin", "Popel"],
     "Development of suitable interatomic potentials for simulation of liquid and amorphous Cu-Zr alloys",
     2009, "89", "967"),
    ("10.1016/0364-5916(78)90011-1", ["Hillert", "Jarl"], "A model for alloying in ferromagnetic metals", 1978, "2", "227"),
    ("10.1016/0378-4363(81)91004-4", ["Inden"], "The role of magnetism in the calculation of phase diagrams", 1981,
     "103", "82"),
    # interatomic potentials of docs/validation.md
    ("10.1103/PhysRevB.33.7983", ["Foiles", "Baskes", "Daw"],
     "Embedded-atom-method functions for the fcc metals Cu, Ag, Au, Ni, Pd, Pt, and their alloys", 1986, "33", "7983"),
    ("10.1103/PhysRevB.63.224106", ["Mishin", "Mehl", "Papaconstantopoulos", "Voter", "Kress"],
     "Structural stability and lattice defects in copper", 2001, "63", "224106"),
    ("10.1016/S1359-6454(01)00287-7", ["Zhou", "Wadley", "Johnson", "Larson", "Tabat", "Cerezo", "Petford-Long", "Smith",
                                       "Clifton", "Martens", "Kelly"],
     "Atomic scale structure of sputtered metal multilayers", 2001, "49", "4005"),
    ("10.1080/14786430310001613264", ["Mendelev", "Han", "Srolovitz", "Ackland", "Sun", "Asta"],
     "Development of new interatomic potentials appropriate for crystalline and liquid iron", 2003, "83", "3977"),
    ("10.1103/PhysRevB.73.024116", ["Sun", "Mendelev", "Becker", "Kudin", "Haxhimali", "Asta", "Hoyt", "Karma", "Srolovitz"],
     "Crystal-melt interfacial free energies in hcp metals: A molecular dynamics study of Mg", 2006, "73", "024116"),
    ("10.1080/09500830701191393", ["Mendelev", "Ackland"],
     "Development of an interatomic potential for the simulation of phase transformations in zirconium", 2007, "87", "349"),
]


def plain(s: str) -> str:
    """Lower-case ASCII without spaces or punctuation, so typography (dashes, quotes, case) does not count."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(c for c in s if c.isalnum())


def get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-refcheck", "Accept": accept})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"no answer from {url}")


def record(identifier: str) -> dict:
    """Authors, title, year, volume and first page from the registry that holds the identifier."""
    if identifier.startswith("arXiv:"):
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={identifier[6:]}", "application/atom+xml"))
        e = root.find("a:entry", ns)
        return {"authors": [a.find("a:name", ns).text.split()[-1] for a in e.findall("a:author", ns)],
                "title": " ".join(e.find("a:title", ns).text.split()), "year": int(e.find("a:published", ns).text[:4]),
                "volume": None, "page": None}
    if identifier.startswith("10.5281/"):
        a = json.loads(get(f"https://api.datacite.org/dois/{identifier}"))["data"]["attributes"]
        # creators come as "Family, Given" or "Given Family"; Zenodo's familyName field is not reliable
        # (pyzentropy's holds the whole name), so the display name is parsed instead
        names = [c["name"].split(",")[0] if "," in c["name"] else c["name"].split()[-1] for c in a["creators"]]
        return {"authors": names,
                "title": a["titles"][0]["title"], "year": int(a["publicationYear"]), "volume": None, "page": None}
    m = json.loads(get(f"https://api.crossref.org/works/{identifier}"))["message"]
    return {"authors": [a.get("family", "") for a in m.get("author", [])], "title": " ".join(m["title"][0].split()),
            "year": m["issued"]["date-parts"][0][0], "volume": m.get("volume"),
            "page": (m.get("page") or m.get("article-number") or "").split("-")[0] or None}


def main() -> int:
    problems = []
    for identifier, authors, title, year, volume, page in REFERENCES:
        got = record(identifier)
        diffs = []
        if [plain(a) for a in got["authors"]] != [plain(a) for a in authors]:
            diffs.append(f"authors {got['authors']}")
        if plain(title) not in plain(got["title"]):
            diffs.append(f"title '{got['title']}'")
        if year is not None and got["year"] != year:
            diffs.append(f"year {got['year']}")
        if volume is not None and str(got["volume"]) != volume:
            diffs.append(f"volume {got['volume']}")
        if page is not None and got["page"] != page:
            diffs.append(f"first page {got['page']}")
        print(f"{'MISMATCH' if diffs else 'ok      '} {identifier} {'; '.join(diffs)}")
        if diffs:
            problems.append(identifier)
        time.sleep(1)
    print(f"{len(REFERENCES)} references checked, {len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
