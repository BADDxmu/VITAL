import os
import pandas as pd

def padding_segment(load_list='example_list', path_load='./fasta_benchmark/'):

    CURRENT_DIR = os.path.dirname(__file__)
    test = pd.read_table(load_list, header=None)

    path_save = os.path.join(CURRENT_DIR, 'tmp/')

    os.makedirs(f'{path_save}/fasta/sequence/', exist_ok=True)
    os.makedirs(f'{path_save}/fasta/fragment/', exist_ok=True)

    window = 10
    step = 5

    for i in range(len(test)):
        id1 = test[0][i]
        id2 = test[1][i]

        seq1 = pd.read_table(f'{path_load}{id1}.fasta').iloc[0,0]
        seq2 = pd.read_table(f'{path_load}{id2}.fasta').iloc[0,0]


        ### padding ###
        # seq1 padding
        len1 = len(seq1)
        if len1 < window:
            padding_length = window - len1
            seq1_padding0 = 'X' * (padding_length // 2)
            seq1_padding = 'X' * (padding_length - len(seq1_padding0))
            seq1 = seq1_padding0 + seq1 + seq1_padding
        else:
            padding_length = (len1-window) % step
            if padding_length:
                seq1_padding = 'X' * (step - padding_length)
                seq1 = seq1 + seq1_padding

        # seq2 padding
        len2 = len(seq2)
        padding_length = (len2-window) % step
        if padding_length:
            seq2_padding = 'X' * (step - padding_length)
            seq2 = seq2 + seq2_padding


        seg1 = []
        seg2 = []
        seq1_seg = []
        seq2_seg = []
        ### sliding window ###
        for i in range(window, len(seq1)+1, step):
            seg1.append(f'{id1}_{i-window+1}_{i}')
            seq1_seg.append(seq1[i-window:i])

        for i in range(window, len(seq2)+1, step):
            seg2.append(f'{id2}_{i-window+1}_{i}')
            seq2_seg.append(seq2[i-window:i])


        ### seq save ###
        df_seq1 = pd.DataFrame({'seq':[f'{id1}'], 'chain':[seq1]})
        df_seq2 = pd.DataFrame({'seq':[f'{id2}'], 'chain':[seq2]})

        ## fasta ##
        df_seq1['seq'] = df_seq1['seq'].apply(lambda x: '>' + x)
        df_seq2['seq'] = df_seq2['seq'].apply(lambda x: '>' + x)
        df_seq1 = df_seq1[['seq','chain']].stack().reset_index(drop=True)
        df_seq2 = df_seq2[['seq','chain']].stack().reset_index(drop=True)

        df_seq1.to_csv(f'{path_save}/fasta/sequence/{id1}.fasta', sep='\t', header=False, index=False)
        df_seq2.to_csv(f'{path_save}/fasta/sequence/{id2}.fasta', sep='\t', header=False, index=False)

        ### seg save ###
        df_seg1 = pd.DataFrame({'seg':seg1, 'chain':seq1_seg})
        df_seg2 = pd.DataFrame({'seg':seg2, 'chain':seq2_seg})

        ## fasta ##
        df_seg1['seg'] = df_seg1['seg'].apply(lambda x: '>' + x)
        df_seg2['seg'] = df_seg2['seg'].apply(lambda x: '>' + x)
        df_seg1 = df_seg1[['seg','chain']].stack().reset_index(drop=True)
        df_seg2 = df_seg2[['seg','chain']].stack().reset_index(drop=True)

        df_seg1.to_csv(f'{path_save}/fasta/fragment/{id1}.fasta', sep='\t', header=False, index=False)
        df_seg2.to_csv(f'{path_save}/fasta/fragment/{id2}.fasta', sep='\t', header=False, index=False)


if __name__ == "__main__":
    padding_segment(load_list='example_list', path_load='./fasta_benchmark/')