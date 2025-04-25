import csv
#lang = ['as','bn','gu','hi','kn','ml','mr','ne','or','pa','ta','te','ur']
lang = ['as','bn']
for l in lang:
# Define the dictionary
    transliteration_dict = {}

    # Path to your TSV file
    tsv_file_path = f"/data/akshantar/{l}/{l}.translit.sampled.test.tsv"

    # Read the TSV file
    with open(tsv_file_path, "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")  # Tab-separated values
        
        for row in tsv_reader:
            if len(row) < 2:
                continue  # Skip invalid rows
            
            indic_word, transliteration = row[0], row[1]
            
            if indic_word in transliteration_dict:
                transliteration_dict[indic_word].append(transliteration)
            else:
                transliteration_dict[indic_word] = [transliteration]

    with open(f'/data/finetune_res1_gemmafull/{l}.txt', "r", encoding="utf-8") as file:
        tsv_reader1 = csv.reader(file, delimiter="\t")
        data1=list(tsv_reader1)
    c=0
    for a,b in data1:
        t=b.lower()
        for x in transliteration_dict[a]:
            if x.lower()==t:
                c=c+1
    print(f"{c / len(data1)} -> {l}")
