import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, ttest_1samp, norm, expon
import glob
import os

def difference_of_means( x1, x2, num_runs=500, tail='both'):
    # x1: (num_samples1,)
    # x2: (num_samples2,)
    #
    # tail: (string) {'both', 'x1>x2', 'x2>x1'}

    mean_actual = np.nanmean(x1) - np.nanmean(x2)

    num_x1_samples = x1.size
    num_x2_samples = x2.size

    means_perm = []
    x = np.concatenate([x1, x2])
    for irun in range(num_runs):

        r = np.random.permutation(x.size)

        x1_sample = x[r[:num_x1_samples]]
        x2_sample = x[r[num_x1_samples:]]

        means_perm.append(np.nanmean(x1_sample) - np.nanmean(x2_sample))

    means_perm = np.array(means_perm)

    pvalue = np.sum(means_perm > np.abs(mean_actual)) + np.sum(means_perm < -np.abs(mean_actual))
    pvalue = pvalue / num_runs

    pvalue = np.clip(pvalue, a_min=1/num_runs, a_max=None)

    return pvalue

np.random.seed(0)
data_v4 = np.load('./V4_neural_data/responses_repeat_averaged/responses_190923.npy')
data_v4_session201025 = np.load('./V4_neural_data/responses_repeat_averaged/responses_201025.npy')
data_v4_session211022 = np.load('./V4_neural_data/responses_repeat_averaged/responses_211022.npy')
data_v4_session210225 = np.load('./V4_neural_data/responses_repeat_averaged/responses_210225.npy')

print(data_v4.shape,data_v4_session201025.shape,  
      data_v4_session211022.shape, data_v4_session210225.shape)

response_dir = f'./resnet_500k_resnet_response/'


def image_responses(response_dir = response_dir, dataset_500k= True, neuron_to_use=0):

    response_files = sorted(glob.glob(os.path.join(response_dir, '500k_responses_batch_*.npy')))
   
    responses = []
    for resp_file in response_files:
        responses.append(np.load(resp_file, mmap_mode='r')) 
    responses = np.concatenate(responses, axis=0)

    return responses



dnn_resp = image_responses(response_dir = response_dir, dataset_500k= True)
data_dnn = dnn_resp.T

skew_values_dnn = [skew(neuron_responses) for neuron_responses in data_dnn]



skew_values_v4_190923 = [skew(neuron_responses) for neuron_responses in data_v4]
skew_values_v4_201025 = [skew(neuron_responses) for neuron_responses in data_v4_session201025]
skew_values_v4_211022 = [skew(neuron_responses) for neuron_responses in data_v4_session211022]
skew_values_v4_210225 = [skew(neuron_responses) for neuron_responses in data_v4_session210225]

skew_values_v4 = skew_values_v4_190923 + skew_values_v4_201025 + skew_values_v4_211022+skew_values_v4_210225 


mean_dnn = np.mean(skew_values_dnn)
mean_neuron = np.mean(skew_values_v4)


skewness1 = skew_values_dnn
skewness2 = skew_values_v4


all_skewness = np.concatenate([skewness1, skewness2])
min_edge = all_skewness.min()
max_edge = all_skewness.max()

n_bins = 30
shared_bins = np.linspace(min_edge, max_edge, n_bins + 1)

print(shared_bins)
plt.figure(figsize=(8, 5))
plt.hist(skewness1, bins=shared_bins, density=True, alpha=0.5, label='DNN 1024 units', color='#43a1c9')
plt.hist(skewness2, bins=shared_bins, density=True, alpha=0.5, label='V4 219 units', color='#aad8b5')
plt.xlim(-1,7)
plt.xlabel("Skewness")

plt.legend()
plt.savefig('bin30.pdf',dpi=100)
plt.tight_layout()
plt.show()
