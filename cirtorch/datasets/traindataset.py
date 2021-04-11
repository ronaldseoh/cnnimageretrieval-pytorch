import os
import pickle
import pdb

import torch
import torch.utils.data as data
import tqdm

from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of 
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader,
                 dense_refresh_batch_and_nearby=-1, dense_refresh_batch_multi_hop=-1, dense_refresh_batch_random=-1):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = os.path.join(db_root, 'ims')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            ## TODO: NOT IMPLEMENTED YET PROPOERLY (WITH AUTOMATIC DOWNLOAD)

            # setting up paths
            db_root = '/mnt/fry2/users/datasets/landmarkscvprw18/recognition/'
            ims_root = os.path.join(db_root, 'images', 'train')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [os.path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]
        else:
            raise(RuntimeError("Unknown dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']

        ## If we want to keep only unique q-p pairs 
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10
        
        # Dense refresh experiments
        self.qvecs = None
        self.poolvecs = None
        self.pvecs = None
        
        self.dense_refresh_batch_and_nearby = dense_refresh_batch_and_nearby
        self.dense_refresh_batch_multi_hop = dense_refresh_batch_multi_hop
        self.dense_refresh_batch_random = dense_refresh_batch_random

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))
        # positive image
        output.append(self.loader(self.images[self.pidxs[index]]))
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

        return output, target, index

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
        
    def extract_query_vectors(self, net, target_data_idxs=[],
                              save_embeds=False,
                              save_embeds_epoch=-1, save_embeds_step=-1, save_embeds_total_steps=-1,
                              save_embeds_path=''):
        # prepare network
        net.cuda()
        
        # if net was in training mode, temporarily switch to eval mode
        was_training = net.training

        if was_training:
            net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():
            if len(target_data_indexes) > 0:
                # Refresh just a single data point specified by target_data_index
                qidxs = [self.qidxs[t] for t in target_data_idxs]
                images_to_rebuild = [self.images[i] for i in qidxs]
            else:
                # Rebuild all queries within the dataset
                qidxs = self.qidxs
                images_to_rebuild = [self.images[i] for i in qidxs]
                
            print('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=images_to_rebuild, imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )

            # extract query vectors
            if self.qvecs is None:
                self.qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()

            for i, input in tqdm.tqdm(enumerate(loader)):
                self.qvecs[:, qidxs[i]] = net(input.cuda()).data.squeeze()

            print('')
            
            # Serialize the query vectors
            if save_embeds:
                print(
                    ">>>>> Epoch {} Step {}/{} query embeddings serialization start.".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))

                torch.save(
                    self.qvecs, os.path.join(save_embeds_path, '{}_queries.pt'.format(save_embeds_step)))
 
                print(
                    ">>>>> Epoch {} Step {}/{} query embeddings serialization complete!".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                    
                print()
                
            # Although not needed in the original training, we need vectors of positive images as well for our purposes.
            self.extract_positive_vectors(
                net, save_embeds, save_embeds_epoch, save_embeds_step, save_embeds_total_steps, save_embeds_path)

        # Restore the training mode
        if was_training:
            net.train()
            
    def extract_negative_pool_vectors(self, net, target_data_idxs=[],
                                      save_embeds=False,
                                      save_embeds_epoch=-1, save_embeds_step=-1, save_embeds_total_steps=-1,
                                      save_embeds_path=''):
        # prepare network
        net.cuda()
        
        # if net was in training mode, temporarily switch to eval mode
        was_training = net.training

        if was_training:
            net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():    
            
            if len(target_data_indexes) > 0:
                # Refresh just a single data point specified by target_data_index
                idxs2images = [self.idxs2images[t] for t in target_data_idxs]
                images_to_rebuild = [self.images[i] for i in idxs2images]
            else:
                # Rebuild all queries within the dataset
                idxs2images = self.idxs2images
                images_to_rebuild = [self.images[i] for i in idxs2images]

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=images_to_rebuild, imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )

            # extract negative pool vectors
            if self.poolvecs is None:
                self.poolvecs = torch.zeros(net.meta['outputdim'], len(self.idxs2images)).cuda()

            for i, input in tqdm.tqdm(enumerate(loader)):
                self.poolvecs[:, idxs2images[i]] = net(input.cuda()).data.squeeze()
            print('')
            
            # Serialize the query vectors
            if save_embeds:
                print(
                    ">>>>> Epoch {} Step {}/{} pool embeddings serialization start.".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                
                torch.save(
                    self.poolvecs, os.path.join(save_embeds_path, '{}_pools.pt'.format(save_embeds_step)))
                    
                print(
                    ">>>>> Epoch {} Step {}/{} pool embeddings serialization complete!".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                    
                print()

        # Restore the training mode
        if was_training:
            net.train()
            
    def extract_positive_vectors(self, net, target_data_idxs=[],
                                 save_embeds=False,
                                 save_embeds_epoch=-1, save_embeds_step=-1, save_embeds_total_steps=-1,
                                 save_embeds_path=''):
        # prepare network
        net.cuda()
        
        # if net was in training mode, temporarily switch to eval mode
        was_training = net.training

        if was_training:
            net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            if len(target_data_indexes) > 0:
                # Refresh just a single data point specified by target_data_index
                pidxs = [self.idxs2images[t] for t in target_data_idxs]
                images_to_rebuild = [self.images[i] for i in pidxs]
            else:
                # Rebuild all positive images within the dataset
                pidxs = self.pidxs
                images_to_rebuild = [self.images[i] for i in pidxs]

            print('>> Extracting descriptors for positive images...')
            # prepare positive image loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract positive image vectors
            if self.pvecs is None:
                self.pvecs = torch.zeros(net.meta['outputdim'], len(self.pidxs)).cuda()

            for i, input in tqdm.tqdm(enumerate(loader)):
                self.pvecs[:, pidxs[i]] = net(input.cuda()).data.squeeze()
            print('')
            
            # Serialize the positive vectors
            if save_embeds:
                print(
                    ">>>>> Epoch {} Step {}/{} positive embeddings serialization start.".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))

                torch.save(
                    self.pvecs, os.path.join(save_embeds_path, '{}_positive.pt'.format(save_embeds_step)))
 
                print(
                    ">>>>> Epoch {} Step {}/{} positive embeddings serialization complete!".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                    
                print()

        # Restore the training mode
        if was_training:
            net.train()


    def create_epoch_tuples(self, net,
                            refresh_positive_pool=True,
                            refresh_negative_pool=True,
                            save_embeds=False,
                            save_embeds_epoch=-1, save_embeds_step=-1, save_embeds_total_steps=-1,
                            save_embeds_path=''):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
        print(">>>> used network: ")
        print(net.meta_repr())

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        if refresh_positive_pairs:
            self.idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
            
            self.qidxs = [self.qpool[i] for i in self.idxs2qpool]
            self.pidxs = [self.ppool[i] for i in self.idxs2qpool]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if refresh_negative_pairs:
            if self.nnum == 0:
                self.nidxs = [[] for _ in range(len(self.qidxs))]
                return 0

            # draw poolsize random images for pool of negatives images
            self.idxs2images = torch.randperm(len(self.images))[:self.poolsize]

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():
            # extract query vectors
            self.extract_query_vectors(
                net, save_embeds, save_embeds_epoch, save_embeds_step, save_embeds_total_steps, save_embeds_path)

            # extract negative pool vectors
            self.extract_negative_pool_vectors(
                net, save_embeds, save_embeds_epoch, save_embeds_step, save_embeds_total_steps, save_embeds_path)

            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(self.poolvecs.t(), self.qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = self.idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if not self.clusters[potential] in clusters:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
                
            if save_embeds:
                # Need to save idxs2images, so that we could figure out the original index
                # of negative examples in poolvecs
                print(
                    ">>>>> Epoch {} Step {}/{} idxs2images serialization start.".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))

                torch.save(
                    self.idxs2images, os.path.join(save_embeds_path, '{}_idxs2images.pt'.format(save_embeds_step)))
 
                print(
                    ">>>>> Epoch {} Step {}/{} idxs2images serialization complete!".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                    
                print()
                
                # self.nidxs
                print(
                    ">>>>> Epoch {} Step {}/{} nidxs serialization start.".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))

                torch.save(
                    self.nidxs, os.path.join(save_embeds_path, '{}_nidxs.pt'.format(save_embeds_step)))
 
                print(
                    ">>>>> Epoch {} Step {}/{} nidxs serialization complete!".format(save_embeds_epoch, save_embeds_step, save_embeds_total_steps))
                    
                print()
                    
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')

        return (avg_ndist/n_ndist).item()  # return average negative l2-distance
