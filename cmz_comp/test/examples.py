from cmz_comp.compressor import Compressor
from cmz_comp.decompressor import Decompressor


DATA_PATH = 'data/'

IMAGES = (
    'lena-gray',
    'baboon-gray',
    'lizard-gray',
)

if __name__ == '__main__':
    for test_img in IMAGES:
        src_path = '{}{}{}'.format(DATA_PATH, test_img, '.png')
        comp_path = '{}{}{}'.format(DATA_PATH, test_img, '.cmz')
        dump_path = '{}{}{}'.format(DATA_PATH, test_img, '.dump.json')
        decomp_path = '{}{}{}'.format(DATA_PATH, test_img, '.decomp.png')
        print('Processing {}'.format(src_path))
        print('Compressing')
        Compressor(
            input_path=src_path, output_path=comp_path,
            dump_history_path=dump_path, dump_history_plots=True
        ).run()
        print('Decompressing')
        Decompressor(input_path=comp_path, output_path=decomp_path).run()
        print('Done')
    print('All done')
