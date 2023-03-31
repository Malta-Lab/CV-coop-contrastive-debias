import torch
import torch.nn as nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
	"""
	Supervised Contrastive Learning loss from:
	https://arxiv.org/pdf/2004.11362.pdf.
	"""
	def __init__(
		self,
		temperature=0.5,
		contrast_mode='all',
		base_temperature=0.5
	):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature  # scales up the loss

	def forward(
		self, 
		features,
		labels=None, 
		mask=None, 
		preds=None
	):
		"""Compute loss for model
		
		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j} = 1
				  if sample j
			has the same class as sample i. Can be asymmetric.
			(We can pass the mask later instead of the labels to compute the
			loss only on the correctly classified samples)
		Returns:
			A loss scalar.
		"""
		device = (
			torch.device('cuda')
			if features.is_cuda
			else torch.device('cpu')
		)

		# L2 normalization on features
		features = F.normalize(features, p=2, dim=1)

		if len(features.shape) == 2:
			features = features.view(features.shape[0], 1, -1)

		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]

		if labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError(
					'Num of labels does not match num of features'
				)
			mask = torch.eq(labels, labels.T).float().to(device)
			# only consider misclassified instances
			if preds is not None:
				preds = torch.argmax(preds, 1)
				mask_incorrect = \
					(labels != preds).unsqueeze(2).repeat(1,1,labels.shape[1])
				mask * mask_incorrect
		else:
			mask = mask.float().to(device)

		# Define features from original images features
		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
	
		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature
		)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)

		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask

		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

		# remove NaN entries
		is_nan = torch.isnan(mean_log_prob_pos)
		mean_log_prob_pos[is_nan] = 0

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()
		return loss


class ReconstructionContrastiveLoss(nn.Module):
	def __init__(self, distance='l2'):
		super().__init__()
		if distance not in ['l2', 'l1', 'max']:
			raise Exception("Distance should be in ['l2', 'l1', 'max']")

		if distance == 'l2':
			self.distance = self.l2
		elif distance == 'l1':
			self.distance = self.l1
		elif distance == 'max':
			self.distance = self.max_distance

	def l2(self, predicted, gt):
		x = torch.pow(predicted - gt, 2)
		x = x.mean(dim=(1, 2, 3))
		return x

	def l1(self, predicted, gt):
		x = torch.abs(predicted - gt)
		x = x.mean(dim=(1, 2, 3))
		return x

	def max_distance(self, predicted, gt):
		return max(
			self.l2(predicted, gt),
			self.l1(predicted, gt)
		)

	def forward(self, features, gt_features, pred, y):
		features = F.normalize(features, p=2, dim=1)

		# Create mask and set the same shape as features		
		# correct = torch.argmax(pred, 1) == y
		# correct = correct.view(correct.shape[0], 1, 1, 1)
		# correct = correct.repeat(1, *features.shape[1:])

		# Create mask and set the same shape as features		
		correct_0 = (torch.argmax(pred, 1) == y) & (y == 0)		#acertou 0
		correct_1 = (torch.argmax(pred, 1) == y) & (y == 1)		#acertou 1
		incorrect_0 = (torch.argmax(pred, 1) != y) & (y == 0) #preveu 1 e é 0
		incorrect_1 = (torch.argmax(pred, 1) != y) & (y == 1) #preveu 0 e é 1

		correct_0 = correct_0.view(correct_0.shape[0], 1, 1, 1)
		correct_0 = correct_0.repeat(1, *features.shape[1:])
		correct_1 = correct_1.view(correct_1.shape[0], 1, 1, 1)
		correct_1 = correct_1.repeat(1, *features.shape[1:])
		incorrect_0 = incorrect_0.view(incorrect_0.shape[0], 1, 1, 1)
		incorrect_0 = incorrect_0.repeat(1, *features.shape[1:])
		incorrect_1 = incorrect_1.view(incorrect_1.shape[0], 1, 1, 1)
		incorrect_1 = incorrect_1.repeat(1, *features.shape[1:])
		
		# Apply mask to features

		inc_features_0 = features * incorrect_0
		inc_features_1 = features * incorrect_1
  
		#get gt_features
		gt_features_0 = gt_features * correct_0
		gt_features_1 = gt_features * correct_1
  
		# TODO está calculando a diferença em relação a 0, como repetir as features do gt para verificarmos as distâncias das features incorretas?
  
		# Calculate distance between features
		difference_0 = self.distance(inc_features_0, gt_features_0)
		difference_1 = self.distance(inc_features_1, gt_features_1)
		difference = difference_0 + difference_1
  
		return difference.mean(dim=0)

		# Apply mask to features
		# features = features * correct
		# gt_features = gt_features * correct

		# Calculate distance between features
		difference = self.distance(features, gt_features)
		return difference.mean(dim=0)