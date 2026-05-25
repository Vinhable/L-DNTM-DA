import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .Encoder import Encoder
from .ETC import ETC
from .UWE import UWE
from .LLMGuider import LLMGuider

class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
                
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)

class LDAR_DTM(nn.Module):
    def __init__(self,
                 vocab_size, num_times, num_topics, train_time_wordfreq,
                 word_embeddings, en_units, dropout, beta_temp,
                 temperature, weight_neg, weight_pos, weight_UWE, neg_topk,
                 plm_model_name='all-mpnet-base-v2',
                 align_warm_up_epoch=10,
                 align_frequency=10,
                 weight_loss_align=1.0,
                 plm_top_k=15,
                 align_sinkhorn_alpha=0.1,
                 evo_warm_up_epoch=10,
                 weight_loss_evo=1.0,
                 llm_warm_up_epochs=150,
                 lambda_contrastive=0.0,
                 krouter_model_name="gpt-5.5",
                 llm_max_workers=20,
                 llm_contrastive_temperature=0.1,
                 llm_guidance_refresh_rate=10,
                 llm_top_k=15,
                 llm_history_length=3,
                 llm_max_retries=3,
                 llm_retry_delay=5,
                 idx_to_word=None,
                 word_to_idx=None,
                 llm_batch_size=5,
                 llm_log_path="./llm_logs/"
                ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_times = num_times
        self.num_topics = num_topics
        self.train_time_wordfreq = train_time_wordfreq

        if idx_to_word is None or word_to_idx is None:
            raise ValueError("idx_to_word and word_to_idx must be provided for LLMGuider.")
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        
        self.align_warm_up_epoch = align_warm_up_epoch
        self.align_frequency = align_frequency
        self.weight_loss_align = weight_loss_align
        self.plm_top_k = plm_top_k
        self.align_sinkhorn_alpha = align_sinkhorn_alpha

        self.evo_warm_up_epoch = evo_warm_up_epoch
        self.weight_loss_evo = weight_loss_evo 
        self.llm_warm_up_epochs = llm_warm_up_epochs

        self.llm_guider = LLMGuider(
            lambda_contrastive=lambda_contrastive,
            krouter_model_name=krouter_model_name,
            llm_max_workers=llm_max_workers,
            llm_contrastive_temperature=llm_contrastive_temperature,
            llm_guidance_refresh_rate=llm_guidance_refresh_rate,
            llm_top_k=llm_top_k,
            llm_history_length=llm_history_length,
            llm_max_retries=llm_max_retries,
            llm_retry_delay=llm_retry_delay,
            num_times=self.num_times,
            num_topic=self.num_topics,
            log_path=llm_log_path,
            llm_batch_size=llm_batch_size        
        )
        
        print(f"Loading PLM model ({plm_model_name}) for alignment...")
        self.plm_model = SentenceTransformer(plm_model_name)
        self.plm_dim = self.plm_model.get_sentence_embedding_dimension()
        self.register_buffer('plm_topic_embeddings', 
                             torch.zeros(self.num_times, self.num_topics, self.plm_dim))

        encoder_args = type('Args', (object,), {
            'vocab_size': vocab_size, 'num_topic': num_topics,
            'model': type('ModelArgs', (object,), {'en1_units': en_units, 'dropout': dropout})
        })()
        self.encoder = Encoder(encoder_args, self.plm_dim)

        self.a = 1 * np.ones((1, self.num_topics)).astype(np.float32)
        mu2 = torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        var2 = torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / self.num_topics))).T + (1.0 / (self.num_topics * self.num_topics)) * np.sum(1.0 / self.a, 1)).T)
        self.register_buffer('mu2', mu2)
        self.register_buffer('var2', var2)

        self.decoder_bn = nn.BatchNorm1d(self.vocab_size, affine=False)
        self.word_embeddings = nn.Parameter(torch.from_numpy(word_embeddings).float())
        self.vae_dim = self.word_embeddings.shape[1]
        self.topic_embeddings = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros(self.num_topics, self.word_embeddings.shape[1]))
            .repeat(self.num_times, 1, 1)
        )
        self.beta_temp = beta_temp
        self.ETC = ETC(self.num_times, temperature, weight_neg, weight_pos)
        self.UWE = UWE(self.ETC, self.num_times, temperature, weight_UWE, neg_topk)

        self.vae_evo_projection = nn.Linear(self.vae_dim, self.plm_dim)
        self.cka_loss_fn = CKALoss()
        
    def get_beta(self):
        dist = self._pairwise_euclidean_dist(
            F.normalize(self.topic_embeddings, dim=-1), 
            F.normalize(self.word_embeddings, dim=-1)
        )
        beta = F.softmax(-dist / self.beta_temp, dim=1)
        return beta

    def _pairwise_euclidean_dist(self, x, y):
        x_sq = torch.sum(x ** 2, axis=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, axis=-1)
        cost = x_sq + y_sq - 2 * torch.matmul(x, y.t())
        return cost.clamp(min=0)

    def get_theta(self, x, doc_embedding, times=None):
        theta, _, _ = self.encoder(x, doc_embedding)
        return theta
    
    def get_KL(self, mu, logvar):
        var = logvar.exp()
        KLD = 0.5 * ((var / self.var2 + (mu - self.mu2)**2 / self.var2 + 
                      self.var2.log() - logvar).sum(axis=1) - self.num_topics)
        return KLD.mean()

    def decode(self, theta, beta_for_docs):
        recon_logits = torch.bmm(theta.unsqueeze(1), beta_for_docs).squeeze(1)
        return F.softmax(self.decoder_bn(recon_logits), dim=-1)
    
    def update_plm_topic_embeddings(self, epoch):
        if epoch % self.align_frequency != 0 or epoch < self.align_warm_up_epoch:
            return

        print(f"\n[Epoch {epoch}] Updating PLM topic embeddings (for L_doc-align & L_evo-align)...")
        
        with torch.no_grad():
            beta = self.get_beta().detach() 

        top_indices = torch.topk(beta, k=self.plm_top_k, dim=-1).indices 
        
        all_topic_strings = []
        for t in range(self.num_times):
            for k in range(self.num_topics):
                words = [self.idx_to_word[idx.item()] for idx in top_indices[t, k]]
                all_topic_strings.append(" ".join(words))

        plm_device = self.topic_embeddings.device
        self.plm_model.to(plm_device)
        
        embeddings = self.plm_model.encode(
            all_topic_strings, 
            show_progress_bar=False, 
            convert_to_tensor=True, 
            device=plm_device
        ) 
        
        self.plm_topic_embeddings.data = embeddings.reshape(
            self.num_times, self.num_topics, -1
        ) 
        print(f"[Epoch {epoch}] PLM topic embeddings updated.")

    
    def sinkhorn_algorithm(self, cost_matrix, epsilon, max_iters=200):
        """
        Giải bài toán OT xấp xỉ bằng thuật toán Sinkhorn-Knopp.
        Cost matrix C: [Batch_size, Num_topics]
        a (doc marginals): uniform 1/Batch_size
        b (topic marginals): uniform 1/Num_topics
        """
        B, K = cost_matrix.shape
        device = cost_matrix.device
        M = torch.exp(-cost_matrix / epsilon)
        u = torch.ones(B, device=device) / B
        v = torch.ones(K, device=device) / K
        a_vec = torch.ones(B, device=device) / B 
        b_vec = torch.ones(K, device=device) / K

        for _ in range(max_iters):
            denominator_a = torch.matmul(M, b_vec) + 1e-10 
            a_vec = u / denominator_a  
            denominator_b = torch.matmul(M.t(), a_vec) + 1e-10
            b_vec = v / denominator_b  

        Pi_star = a_vec.unsqueeze(1) * M * b_vec.unsqueeze(0)
        
        return Pi_star
    
    def forward(self, x, times, doc_embedding, epoch=None):
        # Hybrid Encoder: Truyền cả x và doc_embedding
        theta, mu, logvar = self.encoder(x, doc_embedding)
        kl_theta = self.get_KL(mu, logvar)

        beta = self.get_beta()
        time_index_beta = beta.index_select(0, times.long()) if times.ndim > 0 else beta[times]
        recon_x = self.decode(theta, time_index_beta)
        NLL = -(x * recon_x.log()).sum(axis=1).mean()
        
        loss_ETC = self.ETC(self.topic_embeddings)
        loss_UWE = self.UWE(self.train_time_wordfreq, beta, self.topic_embeddings, self.word_embeddings)

        loss_cfdtm = NLL + kl_theta + loss_ETC + loss_UWE

        loss_align = torch.tensor(0.0, device=x.device)
        if epoch is not None and epoch >= self.align_warm_up_epoch:
            plm_topic_emb_for_docs = self.plm_topic_embeddings.index_select(
                0, times.long()
            ) if times.ndim > 0 else self.plm_topic_embeddings[times]

            diff = doc_embedding.unsqueeze(1) - plm_topic_emb_for_docs
            cost_matrix = torch.sum(diff ** 2, dim=2) 

            pi_star = self.sinkhorn_algorithm(
                cost_matrix, 
                epsilon=self.align_sinkhorn_alpha, 
                max_iters=200
            )

            batch_size = x.size(0)
            theta_prime = batch_size * pi_star
            theta_prime = theta_prime.clamp(min=1e-10) 
            theta_prime = theta_prime / theta_prime.sum(dim=1, keepdim=True)

            loss_align = F.kl_div(
                F.log_softmax(theta, dim=1), 
                theta_prime.detach(),       
                reduction='batchmean'
            )

        loss_evo_align = torch.tensor(0.0, device=x.device)
        if epoch is not None and epoch >= self.evo_warm_up_epoch:
            v_vae = self.topic_embeddings[1:] - self.topic_embeddings[:-1] # [T-1, K, D_vae]
            v_plm = self.plm_topic_embeddings[1:] - self.plm_topic_embeddings[:-1] # [T-1, K, D_plm]
            v_vae_projected = self.vae_evo_projection(v_vae) # [T-1, K, D_plm]
            loss_evo_align = self.cka_loss_fn(v_vae_projected, v_plm.detach())

        loss_llm = torch.tensor(0.0, device=x.device)
        if epoch is not None and epoch > self.llm_warm_up_epochs:
            self.llm_guider.update_guidance_cache(epoch, beta.detach(), self.idx_to_word)
            loss_llm = self.llm_guider.calculate_contrastive_loss(self.topic_embeddings, self.word_embeddings, self.word_to_idx)

        total_loss = (
            loss_cfdtm 
            + self.weight_loss_align * loss_align 
            + self.weight_loss_evo * loss_evo_align
            + self.llm_guider.lambda_contrastive * loss_llm
        )

        rst_dict = {
            'loss': total_loss,
            'loss_core': loss_cfdtm,
            'loss_align': loss_align,         
            'loss_evo': loss_evo_align, 
            'loss_llm': loss_llm, 
            'core_nll': NLL,
            'core_kl_theta': kl_theta,
            'core_etc': loss_ETC,
            'core_uwe': loss_UWE,
        }
        return rst_dict