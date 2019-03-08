import glob
import itertools
import os
import re
import tarfile
import urllib.request
from collections import Counter
from io import BytesIO

from torch.utils.data import Dataset


class McGillBillboard(Dataset):
    base = 'McGill-Billboard'
    url = 'https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1'
    filename = 'mcgill-billboard.pkl'

    def __init__(self, root='data/', download=True):
        super(McGillBillboard, self).__init__()
        self.root = root
        self.data_root = os.path.join(self.root, self.base)

        if self.check_integrity():
            print('Dataset already downloaded and checked')
        elif download:
            self.download()
        else:
            raise Exception('Dataset not found or corrupted, use download=True to download.')

        self.preprocess()

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)

    def check_integrity(self):
        if not os.path.exists(self.data_root):
            return False

        songs = os.listdir(self.data_root)
        if len(songs) != 890:
            return False

        return True

    def download(self):
        print('Downloading McGill-Billboard dataset ...')
        os.makedirs(self.root, exist_ok=True)
        tgz_file = BytesIO(urllib.request.urlopen(self.url).read())
        tarfile.open(fileobj=tgz_file).extractall(path=self.root)
        assert self.check_integrity(), "Error downloading or extracting data"

    def preprocess(self):
        x = os.path.join(self.data_root, '**', '*.txt')
        x = glob.glob(x)
        x = sorted(x)
        x = map(self._process_salami, x)
        
        self.info, X, y = zip(*x)
        print('Songs loaded:', len(self.info))

        # for h, yi in zip(self.info, y):
        #     if 'secondarytheme' in yi:
        #         print(h['url'])

        ally = itertools.chain.from_iterable(y)
        allX = itertools.chain.from_iterable(X)
        # print(Counter(ally))
        
        self.chords = sorted(list(set(allX)))
        self.labels = sorted(list(set(ally)))
        
        self.chord2idx = {c: i for i, c in enumerate(self.chords)}
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        
        X = map(lambda chord_seq: [self.chord2idx[c] for c in chord_seq], X)
        y = map(lambda label_seq: [self.label2idx[l] for l in label_seq], y)
        
        self.X = list(X)
        self.y = list(y)
        
    @staticmethod
    def _process_salami(fname):
        with open(fname, 'r') as x:
            salami = x.read()
            lines = salami.split('\n')
            
            # song info
            headers = filter(lambda x: x.startswith('#'), lines)
            headers = map(lambda x: x.lstrip('# ').split(':', maxsplit=1), headers)
            headers = {k: v.strip() for (k, v) in headers}
            headers['url'] = fname

            # print('{title} - {artist} - {metre} - {tonic} ({url})'.format(**headers))
            # print('--------------------------------------')
                        
            ## section regex
            # ^.*\t                    -> timestamp at the beginning of the line
            # (?:[A-Z]'*, )?           -> a section letter, if any, e.g. A'
            # (?P<label>[a-z\-\(\) ]+) -> the section label
            # (?:.*\n)+?               -> lazily matches the rest of the section
            # (?=^.*\t(?:[A-Z]'*, )?(?:[a-z\-]+)) -> stop when a new section starts
            section_regex = r"^.*\t(?:[A-Z]'*, )?(?P<type>[a-z\-]+),(?:.*\n)*?(?=^.*\t(?:[A-Z]'*, )?(?:[a-z\-]+))"
            
            sections = re.finditer(section_regex, salami, re.MULTILINE)
            sections = map(lambda m: (m.groups()[0], m.group()), sections)
            labels, sections = zip(*sections)
            
            def clean_labels(label):
                label = label.replace(r'-', '').replace(' ', '')
                # e.g. chorus-1 -> chorus
                for prefix in ('trans', 'verse', 'chorus', 'prechorus', 'intro', 'instrumental', 'spoken'):
                    label = prefix if label.startswith(prefix) else label
                # modulation -> keychange
                label = 'keychange' if label == 'modulation' else label
                
                return label

            labels = map(clean_labels, labels)
            
            sections = map(lambda x: x.split('\n'), sections)
            
            # a valid symbol is: a chord, *, &pause, x4, N
            valid_symbol = re.compile(r'(?:[A-G][#b]?:\S+)|\*|(?:\&pause)|(?:x\d+)|N')
            # extract valid symbols for each line
            chords = map(lambda x: map(valid_symbol.findall, x), sections)  
            
            def valid_lines(symbols):  # remove lines with no symbols
                return len(symbols) > 0
                
            chords = map(lambda x: filter(valid_lines, x), chords)
            
            # resolve e.g. x4 to repeat the line 4 times
            def resolve_repeats(x):  # XXX only end-of-line repeats are supported
                if x[-1].startswith('x'):
                    reps = int(x[-1][1:])
                    x = x[:-1] * reps
                return x

            chords = map(lambda x: map(resolve_repeats, x), chords)
            chords = map(itertools.chain.from_iterable, chords)
            chords = map(list, chords)
            
            lab_n_chords = zip(labels, chords)
            lab_n_chords = filter(lambda x: x[0] != '', lab_n_chords)
            labels, chords = zip(*lab_n_chords)

            label_seq = []
            chord_seq = []

            for label, chord_sec in zip(labels, chords):
                label_seq.extend([label, ] * (len(chord_sec) - 1))
                label_seq.extend(['eos', ])
                chord_seq.extend(chord_sec)
                
            return headers, chord_seq, label_seq


if __name__ == '__main__':
    d = McGillBillboard()
    # print(McGillBillboard._process_salami('data/McGill-Billboard/0006/salami_chords.txt'))

    print(d.info[2])
    print(d[2])
    

