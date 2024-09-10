import torch

class ProteinNode:
    def __init__(self,ProteinID,ProteinLen,PrimarySeq,SecondarySeq,PHI,PSI):
        self.PDB_ID=ProteinID
        self.ProteinLen=ProteinLen
        self.PrimarySeq=PrimarySeq
        self.SecondarySeq=SecondarySeq       
        self.PHI=PHI
        self.PSI=PSI               
        self.feats=None

def load_FastaFile(FastaFile,IsESM=True, With_SS=True):
    f=open(FastaFile,'r')
    if IsESM:
        AA_Dict={ 'L':4, 'A':5, 'G':6, 'V':7, 'S':8, 'E':9, 'R':10, 'T':11, 'I':12, 'D':13, 'P':14, 'K':15, 'Q':16, 'N':17, 'F':18, 'Y':19, 'M':20, 'H':21, 'W':22,
                  'C':23, 'X':24, 'B':25, 'U':26, 'Z':27, 'O':28}
    else:
        #ProtTrans          
        Standard_AAS=['A', 'L', 'G', 'V', 'S', 'R', 'E', 'D', 'T', 'I', 'P','K','F','Q', 'N', 'Y', 'M', 'H',  'W', 'C']
        AA_Dict={}
        Non_Standard_AAS=list('BJOUXZ')
        for i,AA in enumerate(Standard_AAS):
            AA_Dict.update({AA:i+3})
        for i,AA in enumerate(Non_Standard_AAS):
            AA_Dict.update({AA:23}) 
    Data=[]
    SS=['L','B','E','G','I','H','S','T','P','X']
    SS9_Dict=dict(zip(SS,range(len(SS))))
    SS9_Dict['X']=-100
    SS9_Dict['C']=0
    while True:       
        PDBID=f.readline().strip()
        if len(PDBID)==0:
            break        
        info=PDBID.split()
        ProteinID=info[0][1:]
        PrimarySeq=f.readline().strip()        
        PrimarySeq=PrimarySeq.upper()
        ProteinLen=len(PrimarySeq)
        PrimarySeq=[AA_Dict[e] for e in PrimarySeq]
        PrimarySeq=torch.tensor(PrimarySeq,dtype=torch.long)
        if With_SS:
            SecondarySeq=f.readline().strip()
            SecondarySeq=[SS9_Dict[e] for e in SecondarySeq]
            SecondarySeq=torch.tensor(SecondarySeq,dtype=torch.long)
        else:
            SecondarySeq=None               
        PHI=f.readline().strip().split()
        tmp_list=[]
        for i in range(ProteinLen):
            if PHI[i]!='X' and PHI[i][:3]!='360':
                tmp_list.append(float(PHI[i]))
            else:
                tmp_list.append(float('nan'))
        PHI=torch.tensor(tmp_list,dtype=torch.float)
        PSI=f.readline().strip().split()
        tmp_list=[]
        for i in range(ProteinLen):
            if PSI[i]!='X' and PSI[i][:3]!='360':
                tmp_list.append(float(PSI[i]))
            else:
                tmp_list.append(float('nan'))
        PSI=torch.tensor(tmp_list,dtype=torch.float)       
                 
        Node=ProteinNode(ProteinID,ProteinLen,PrimarySeq,SecondarySeq,PHI/180.0,PSI/180.0)
        Data.append(Node)
    f.close()
    return Data



def load_train_valid_test(IsESM=True):  
   train_list=load_FastaFile('data/spot_1d_train.txt',IsESM)
   valid_list=load_FastaFile('data/spot_1d_valid.txt',IsESM)
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)
   Test2016_list=load_FastaFile('data/TEST2016.txt',IsESM)
   Test2016_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)   
   Test2018_list=load_FastaFile('data/TEST2018.txt',IsESM)
   Test2018_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)   
   return train_list,valid_list,Test2016_list,Test2018_list


def load_four_testsets(IsESM=True):
   Test2020_HQ_list=load_FastaFile('data/TEST2020_HQ.txt',IsESM,False)
   Test2020_HQ_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)   
   CASP12_list=load_FastaFile('data/CASP12.txt',IsESM,False)
   CASP12_list.sort(key=lambda Node: Node.ProteinLen,reverse=False) 
   CASP13_list=load_FastaFile('data/CASP13.txt',IsESM,False)
   CASP13_list.sort(key=lambda Node: Node.ProteinLen,reverse=False) 
   CASPFM_list=load_FastaFile('data/CASPFM.txt',IsESM,False)
   CASPFM_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)    
   return Test2020_HQ_list, CASP12_list, CASP13_list, CASPFM_list




