import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
from torch.optim.lr_scheduler import StepLR
from topmost.utils import _utils
from topmost.utils.logger import Logger

logger = Logger("WARNING")

class DynamicTrainer:
    def __init__(self,
                model,
                dataset,
                num_top_words=15,
                epochs=200,
                learning_rate=0.002,
                batch_size=200,
                lr_scheduler=None,
                lr_step_size=125,
                log_interval=5,
                verbose=False
            ):

        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            logger.info("using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset.train_dataloader.dataset)
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.update_plm_topic_embeddings(epoch)
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset.train_dataloader:

                rst_dict = self.model(
                    x=batch_data['bow'],
                    times=batch_data['times'],
                    doc_embedding=batch_data['doc_embedding'],
                    epoch=epoch
                )

                batch_loss = rst_dict['loss']
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                current_batch_size = batch_data['bow'].size(0) 
                
                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key].item() * current_batch_size

            if self.lr_scheduler:
                lr_scheduler.step()

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                logger.info(output_log)

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_bow, self.dataset.train_times, self.dataset.train_doc_emb)

        return top_words, train_theta

    def test(self, bow, times, doc_emb):
        data_size = bow.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_theta = self.model.get_theta(bow[idx], doc_emb[idx], times[idx])
                theta.extend(batch_theta.cpu().tolist())
        return np.asarray(theta)
    
    def get_beta(self):
        self.model.eval()
        with torch.no_grad():
            beta_tensor = self.model.get_beta()
        return beta_tensor.detach().cpu().numpy()
    
    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words

        print(f"\nGenerating final top words (n={num_top_words}) using refined cache...")
        beta = self.get_beta()

        refined_cache = {}
        if hasattr(self.model, 'llm_guider') and self.model.llm_guider is not None:
             refined_cache = self.model.llm_guider.refined_top_words_cache

        final_top_words_list = []
        for t in range(beta.shape[0]):
            topics_at_time_t = []
            for k in range(beta.shape[1]):
                topic_id = (t, k)
                
                if topic_id in refined_cache:
                    refined_list = refined_cache[topic_id]
                    topics_at_time_t.append(" ".join(refined_list[:num_top_words]))
                else:
                    top_indices = np.argsort(beta[t, k, :])[-num_top_words:][::-1]
                    fallback_words = [self.dataset.vocab[i] for i in top_indices]
                    topics_at_time_t.append(" ".join(fallback_words))
            
            final_top_words_list.append(topics_at_time_t)
            
        return final_top_words_list
    
    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow, self.dataset.train_times, self.dataset.train_doc_emb)
        test_theta = self.test(self.dataset.test_bow, self.dataset.test_times, self.dataset.test_doc_emb)
        return train_theta, test_theta
