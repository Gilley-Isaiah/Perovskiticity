import os
import requests
import time
from unicodedata import normalize


def citation(doi, directory=r"C:\Users\isaia\OneDrive - Northwestern University\Desktop\Perovskites Database\RIS Files"):
    headers = {
        'User-Agent': 'Isaiah Gilley (mailto:gilley@u.northwestern.edu)',
        'Accept': 'application/x-research-info-systems',
    }

    doistring = "blank"
    try:
        response = requests.get(doi, headers=headers, timeout=15)
        ris = response.text
        for line in ris.split('\n'):
            if line.startswith('DO'):
                doistring = line.split(" ")[3].replace("/", "-")
                continue
        rislist = ris.split("\n")
        for position, item in enumerate(rislist):
            if "<scp>" in item:
                rislist[position] = rislist[position].replace("<scp>", "")
                rislist[position] = rislist[position].replace("</scp>", "")
            if "<i>" in item:
                rislist[position] = rislist[position].replace("<i>", "")
                rislist[position] = rislist[position].replace("</i>", "")
            if "<sub>" in item:
                rislist[position] = rislist[position].replace("<sub>", "")
                rislist[position] = rislist[position].replace("</sub>", "")
            if "<sup>" in item:
                rislist[position] = rislist[position].replace("<sup>", "")
                rislist[position] = rislist[position].replace("</sup>", "")

        if doistring == "blank":
            print(f"Server connection established for {doi}, but no RIS information was returned by the "
                  f"server.")
            return

        for forbiddencharacter in ["\\", r"/", ":", "*", "?", '"', "<", ">", "|", "\\\\", r"\\"]:
            doistring = doistring.replace(forbiddencharacter, "")
        with open(directory + '\\' + doistring + '.ris', 'w', encoding='utf-8') as file:
            for line in rislist:
                normalizedline = normalize('NFKD', line).encode('utf8', 'ignore').decode('utf8')
                file.write(normalizedline)
                file.write("\n")
        file.close()
        print(f"Finished with {doistring}")
    except requests.exceptions.Timeout:
        print(f"DOI {doi} timed out without server response.")
    except OSError:
        print(f"DOI {doi} failed. Check if something is wrong with the string.")


def citation_machine(database_path,
                     citation_directory=r"C:\Users\isaia\OneDrive - Northwestern University\Desktop\Perovskites Database\RIS Files"):
    # Make a list of the DOIs to process
    doiset = set()
    with open(database_path, 'r') as file:
        for line in file:
            try:
                doi = line.split("\t")[6]
                if doi.startswith("http:"):
                    doiset.add(doi)
            except IndexError:
                continue
    file.close()

    # First let's check which ones have already been done
    prior_citations = set()
    for path in os.listdir(citation_directory):
        if os.path.isfile(os.path.join(citation_directory, path)) and path.endswith(".ris"):
            prior_citations.add(path)

    # Now we need to only request info for the ones that aren't in the 'prior_citations,' but this requires some formatting.
    # The DOI server will shut us out if we make too many calls too quickly, so we wait 3 seconds between each call.
    count = 0
    for doi in doiset:
        doistring = doi[18:]
        doistring = doistring.replace("/", "-")
        for forbiddencharacter in ["\\", r"/", ":", "*", "?", '"', "<", ">", "|", "\\\\", r"\\"]:
            doistring = doistring.replace(forbiddencharacter, "")
        doistring = doistring + ".ris"
        if doistring.lower() not in prior_citations:
            citation(doi, directory=citation_directory)
            time.sleep(3)
        elif doistring in prior_citations:
            print(f"{doi} is already in the destination folder; skipping.")
        count += 1
        print(f"({count}/{len(doiset)}---{round((count / len(doiset)) * 100, 2)}%)")

    print("Finished exporting all citations.")