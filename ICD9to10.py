# Adapated from https://github.com/bhanratt/ICD9CMtoICD10CM
import sys

class ICD9to10():
    def __init__(self, icd_dict) -> None:
        super().__init__()
        #Import ICD 9 to 10 conversion dictionary
        self.data_table = {}
        try:
            f1=open(icd_dict)
            header=f1.readline()
            for line in f1:
                nine = str.strip(line.split('|')[0])
                ten = str.strip(line.split('|')[1])
                desc = str.strip(line.split('|')[2])
                self.data_table[nine] = [ten,desc]
        #Missing dictionary handling
        except FileNotFoundError:
            print("Missing dependency: icd9to10dictionary.txt")
            sys.exit(1)

    def convert(self, icd9s):
        icd10s = []
        count = 0
        for icd9 in icd9s:
            stripped = str.strip(icd9)
            if stripped in self.data_table:
                count+=1
                icd10s.append(self.data_table[stripped][0])
            else:
                icd10s.append(None)

        # print('Matched '+str(count)+' codes from your list of '+str(len(icd9s))+' codes')
        return icd10s

