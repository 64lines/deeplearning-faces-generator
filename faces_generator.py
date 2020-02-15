from gan_utils import load_faces
from gan_utils import savez_compressed
from gan_utils import plot_faces
from gan_utils import load
from gan_utils import define_composite
from gan_utils import define_generator
from gan_utils import define_discriminator
from gan_utils import load_real_samples
from gan_utils import train

class FacesGenerator(object):

    def __init__(self, example_faces_dir='img_align_celeba/', n_faces=50000):
        self.directory = example_faces_dir
        self.n_faces = n_faces

    def extract_faces(self, dataset_file_name='imgs_128.npz'):
        # load and extract all faces
        all_faces = load_faces(self.directory, self.n_faces)
        print('Loaded: ', all_faces.shape)
        # save in compressed format
        savez_compressed(dataset_file_name, all_faces)
    
    def plot_faces(self, dataset_file_name='imgs_128.npz'):
        data = load(dataset_file_name)
        faces = data['arr_0']
        print('Loaded: ', faces.shape)
        plot_faces(faces, 4)
    
    def generate_faces(self, dataset_file_name='imgs_128.npz'):
        n_blocks = 6
        # size of the latent space
        latent_dim = 100
        # define models
        d_models = define_discriminator(n_blocks)
        # define models
        g_models = define_generator(latent_dim, n_blocks)
        # define composite models
        gan_models = define_composite(d_models, g_models)
        # load image data
        dataset = load_real_samples(dataset_file_name)
        print('Loaded', dataset.shape)
        # train model
        n_batch = [16, 16, 16, 8, 4, 4]
        # 10 epochs == 500K images per training phase
        n_epochs = [5, 8, 8, 10, 10, 10]
        train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
