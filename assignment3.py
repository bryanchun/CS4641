# print(__import__('sklearn').__version__)
# Require 0.20 for GaussianMixture.fit_predict

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

# Dimensionalisty Reduction algorithms
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection

# Neural Network task
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Evaluation Metrics
from sklearn import metrics
from scipy.stats import mode
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

# Plotting tools
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.patches import Ellipse

# Utilities
from timeit import default_timer as timer
import numpy as np
np.random.seed(0)

# Datasets
from sklearn.datasets import load_digits, load_iris

# Preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import sklearn.utils as utils


class Dataset:
  
  def __init__(self, data=None, target=None, C=0, F=0, N=0, name='', target_names=[]):
    
    self.data = data
    self.target = target
    self.C = C
    self.F = F
    self.N = N
    self.name = name
    self.target_names = target_names

        
class Solver:
    
  def __init__(self):
        
    digits = load_digits()
    self.digits = Dataset()
    self.digits.name = 'Digits'
    self.digits.data = digits.data #scale(digits.data)
    self.digits.target = digits.target
    self.digits.C = len(np.unique(self.digits.target))
#         self.digits = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
    self.digits.target_names = digits.target_names
    self.digits.N, self.digits.F = digits.data.shape

    iris = load_iris()
    self.iris = Dataset()
    self.iris.name = 'Iris'
    self.iris.data = iris.data #scale(iris.data)
    self.iris.target = iris.target
    self.iris.C = len(np.unique(self.iris.target))
#         self.iris = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
    self.iris.target_names = iris.target_names
    self.iris.N, self.iris.F = iris.data.shape


    #df = pd.read_csv('data/spotify_classification.csv')
    #y_key = 'target'
    #excluded_keys = [y_key, # y-targets
    #                 'Unnamed: 0', # id
    #                 'mode', # unknown
    #                 'song_title', 'artist' # text data not handled
    #                ]
    #self.spotify = self.Dataset()
    #self.spotify.data = df.drop(columns=excluded_keys)
    #self.spotify.target = df[y_key]
    #self.spotify.C = 2
    #self.spotify.target_names = [0, 1]
        
solver = Solver()


class Plots:
  
  @staticmethod
  def conf_mat(mat, dataset, title):
    """
    Parameters
    
    - mat - confusion matrix
    - dataset : Dataset
    - title : String
    
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap=sns.diverging_palette(220, 20, as_cmap=True),
                xticklabels=dataset.target_names,
                yticklabels=dataset.target_names)
    plt.title(title)
    plt.xlabel('True Label')
    plt.ylabel('Cluster-predicted Label')
    plt.show()

  @staticmethod
  def vis_clusters(data, dataset, hypos, elapsed, method_desc=''):
    """
    Parametetr
    
    - data - transformed data
    - dataset : Dataset
    
    """
    pca = PCA(2)
    projected = pca.fit_transform(data)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=hypos, edgecolor='none', alpha=0.7,
            cmap=plt.cm.tab10)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(f"Visualising PCA-projected space with first 2 components of\n{'' if method_desc == '' else ' '}{method_desc} {dataset.name} Dataset\nfitting took {'%.2f' % elapsed}s")
    plt.colorbar()
    plt.show()
    
  @staticmethod
  def vis_density(data, dataset, clusterer, elapsed, method_desc=''):

    X, gmm = PCA(2).fit_transform(data), clusterer
    
    def draw_ellipse(position, covariance, ax=None, **kwargs):
      """Draw an ellipse with a given position and covariance"""
      ax = ax or plt.gca()

      # Convert covariance to principal axes
      if covariance.shape == (2, 2):
          U, s, Vt = np.linalg.svd(covariance)
          angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
          width, height = 2 * np.sqrt(s)
      else:
          # print(np.sqrt(covariance), np.sqrt(covariance).shape)
          angle = 0
          width, height = 2 * np.sqrt(covariance)

      # Draw the Ellipse
      for nsig in range(1, 4):
          ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                               angle, **kwargs))

    ax = plt.gca()
    labels = gmm.fit(X).predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
      draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(f"Visualising PCA-projected space with first 2 components of\n{'' if method_desc == '' else ' '}{method_desc} {dataset.name} Dataset\nfitting took {'%.2f' % elapsed}s")
    # plt.colorbar()
    plt.show()
    
  @staticmethod
  def vis_components(components, dataset, dimreducer, title=False, order=None):
    components_ = dimreducer.components_ if order == None else dimreducer.components_[order]
    centers = dimreducer.components_.reshape(components, 8, 8)
    fig, ax = plt.subplots(components // 8, 8, figsize=(6, 3))
    for axi, center in zip(ax.flat, centers):
      axi.set(xticks=[], yticks=[])
      axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    if title:
      plt.title(f'Visualising pixelwise components\non {dataset.name} Dataset')
    plt.show()
        
    
    
def kmeans(dataset, main=True,
           confmat=False, vis=False, n_init=100, method_desc='',
           return_acc_v=False, return_aug_data=False):
  """
  Parameters
  
  - dataset : Dataset
  - confmat : Boolean
  
  """
  if main:
    start = timer()

    clusterer = KMeans(n_clusters=dataset.C, n_init=n_init, random_state=0)
    # print(dataset.data.shape)
    clusters = clusterer.fit_predict(dataset.data)
    transformed = clusterer.fit_transform(dataset.data)
    # print(transformed.shape)
    hypos = np.zeros_like(clusters)
    for i in range(dataset.C):
      mask = (clusters == i)
      hypos[mask] = mode(dataset.target[mask])[0]

    elapsed = timer() - start
    print(f"Fitting took {'%.2f' % elapsed}s")

    if confmat:
      mat = metrics.confusion_matrix(dataset.target, hypos)
      acc = metrics.accuracy_score(dataset.target, hypos)
      Plots.conf_mat(mat, dataset,
                    f"Confusion Matrix of cluster-predicted labels\n against true labels for {dataset.name} Dataset\nKMeans\naccuracy_score = {'%.2f' % acc}")

    if vis:
      Plots.vis_clusters(transformed, dataset, hypos, elapsed, method_desc=method_desc)
      
    if return_acc_v:
      acc = metrics.accuracy_score(dataset.target, hypos)
      v = metrics.v_measure_score(dataset.target, hypos)
      return acc, v
    
    if return_aug_data:
      return np.hstack([dataset.data, np.array(hypos)[:, np.newaxis]])
      

def em(dataset, main=True,
       confmat=False, vis=False, n_init=100, covariance_type='full', comp_covariance_type=False, method_desc='',
       return_acc_v=False, return_aug_data=False):
  """
  Parameters
  
  - dataset : Dataset
  - confmat : Boolean
  - covariance_type : String in [{‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}] 
  
  """
  if main:
    start = timer()
    
    clusterer = GMM(n_components=dataset.C, n_init=n_init, random_state=0, covariance_type=covariance_type)
    # print(dataset.data.shape)
    clusters = clusterer.fit_predict(dataset.data)
    clusterer.fit(dataset.data)
    # print(clusterer.covariances_, clusterer.covariances_.shape)
    transformed = clusterer.sample(dataset.N)
    # print(np.array(transformed[0]).shape)
    hypos = np.zeros_like(clusters)
    for i in range(dataset.C):
      mask = (clusters == i)
      hypos[mask] = mode(dataset.target[mask])[0]
    
    elapsed = timer() - start
    print(f"Fitting took {'%.2f' % elapsed}s")
    
    if confmat:
      mat = metrics.confusion_matrix(dataset.target, hypos)
      acc = metrics.accuracy_score(dataset.target, hypos)
      Plots.conf_mat(mat, dataset,
                     f"Confusion Matrix of cluster-predicted labels\n against true labels for {dataset.name} Dataset\nGMM with Covariance type {covariance_type}\naccuracy_score = {'%.2f' % acc}")
      
    if vis:
      # Plots.vis_clusters(transformed[0], dataset, hypos)
      Plots.vis_density(transformed[0], dataset, clusterer, elapsed, method_desc=method_desc)
      
    if return_acc_v:
      acc = metrics.accuracy_score(dataset.target, hypos)
      v = metrics.v_measure_score(dataset.target, hypos)
      return acc, v
    
    if return_aug_data:
      return np.hstack([dataset.data, np.array(hypos)[:, np.newaxis]])
    
  if comp_covariance_type:
    covars = ['full', 'tied', 'diag', 'spherical']
    accs = []
    for covar in covars:
      clusterer = GMM(n_components=dataset.C, n_init=n_init, random_state=0, covariance_type=covar)
      clusters = clusterer.fit_predict(dataset.data)
      hypos = np.zeros_like(clusters)
      for i in range(dataset.C):
        mask = (clusters == i)
        hypos[mask] = mode(dataset.target[mask])[0]
      
      accs.append( metrics.accuracy_score(dataset.target, hypos) )
      
    y_pos = np.arange(len(covars))
    plt.bar(y_pos, accs, align='center')
    plt.xticks(y_pos, covars)
    plt.ylabel('accuracy')
    plt.title(f"GMM cluster-label accuracy across Covariance types\non {dataset.name} Dataset")
    plt.show()
  
  
def pca(dataset, components, main=True,
        use_components_percent=False, vis_components=False, eigen=False, comp_components=False,
       return_dimreduced=False, return_loss=False):
  if main:
    if use_components_percent:
      _components = components
      components = int(dataset.F * components)
      # print(f"{_components} of components are {components} components")

    dimreducer = PCA(n_components=components, whiten=True)
    dimreducer.fit(dataset.data, y=dataset.target)
    transformed = dimreducer.transform(dataset.data)
    reconstructed = dimreducer.inverse_transform(transformed)
    loss = ((dataset.data - reconstructed)**2).mean()
    eigenvalues = dimreducer.explained_variance_

    if return_dimreduced:
      return transformed
    
    if return_loss:
      return loss

    # print(f"Reconstruction loss: {loss}")
    # print(f"Components are this many: {len(dimreducer.components_)}")

    if eigen:
      plt.bar(np.arange(components), eigenvalues)
      plt.xlabel("rank of components")
      plt.ylabel("eigenvalue")
      plt.title(f"Distribution of Eigenvalues on {dataset.name} Dataset")
      plt.show()

    if vis_components:
      print(f"{_components} of components are {components} components")
      Plots.vis_components(components, dataset, dimreducer)
      
  if comp_components:
    plt.subplots(5, 1, figsize=(3, 3))
    cs = [10, 20, 30, 40, 50, 60]
    for c in cs:
      dimreducer = PCA(n_components=c, whiten=True)
      dimreducer.fit(dataset.data, y=dataset.target)
      Plots.vis_components(c, dataset, dimreducer)
    plt.title(f'PCA Components visualisation: {cs}')


def ica(dataset, components, main=True,
        use_components_percent=False, vis_components=False, kurto=False, comp_components=False,
       return_dimreduced=False):
  
  if main:
    if use_components_percent:
      _components = components
      components = int(dataset.F * components)
      # print(f"{_components} of components are {components} components")
      
    dimreducer = FastICA(n_components=components, whiten=True, max_iter=5000, tol=1)
    dimreducer.fit(dataset.data, y=dataset.target)
    transformed = dimreducer.transform(dataset.data)
    kurts = kurtosis(transformed)
    order = np.flip(np.argsort(np.absolute(kurts - 3)))
    # transformed = transformed[order]
    
    if return_dimreduced:
      return transformed

    # print(f"Kurtosis is: {kurt}")
    # print(f"Components are this many: {len(dimreducer.components_)}")

    if kurto:
      plt.bar(np.arange(components), kurts[order] - 3)
      plt.xlabel("components")
      plt.ylabel("kurtosis - 3")
      plt.title(f"Distribution of Kurtosis sorted absolutely\non {dataset.name} Dataset")
      plt.show()

    if vis_components:
      print(f"{_components} of components are {components} components")
      Plots.vis_components(components, dataset, dimreducer)
        
  if comp_components:
    plt.subplots(5, 1, figsize=(3, 3))
    cs = [10, 20, 30, 40, 50, 60]
    for c in cs:
      dimreducer = FastICA(n_components=c, whiten=True, max_iter=5000, tol=1)
      dimreducer.fit(dataset.data, y=dataset.target)
      transformed = dimreducer.transform(dataset.data)
      kurts = kurtosis(transformed)
      order = np.flip(np.argsort(np.absolute(kurts - 3)))
      Plots.vis_components(c, dataset, dimreducer, order=order)
    plt.title(f'ICA Components visualisation: {cs}')
        
   # TODO: Vis components=[10, 20, 30, 64]
        

def rp(dataset, components, main=True,
       use_components_percent=False, vis_components=False, comp_runs=False, reruns=10, cs=[],
      return_dimreduced=False, return_loss=False):
  if use_components_percent:
    _components = components
    components = int(dataset.F * components)
  
  if main:
    dimreducer = GaussianRandomProjection(n_components=components, random_state=0)
    dimreducer.fit(dataset.data, y=dataset.target)
    transformed = dimreducer.transform(dataset.data)
    pinversed_components = np.linalg.pinv(dimreducer.components_)
    reconstructed = utils.extmath.safe_sparse_dot(transformed, pinversed_components.T)
    loss = ((dataset.data - reconstructed)**2).mean()

    if return_dimreduced:
      return transformed
    
    if return_loss:
      return loss
    
    if vis_components:
      print(f"{_components} of components are {components} components")
      Plots.vis_components(components, dataset, dimreducer)
      
    
  if comp_runs:
    normmeans, lossmeans = np.array([]), np.array([])
    normvars, lossvars = np.array([]), np.array([])
    for c in cs:
      norms, losses = np.array([]), np.array([])
      # print(c)
      for _ in range(reruns):
        dimreducer = GaussianRandomProjection(n_components=c)
        dimreducer.fit(dataset.data, y=dataset.target)
        transformed = dimreducer.transform(dataset.data)
        pinversed_components = np.linalg.pinv(dimreducer.components_)
        reconstructed = utils.extmath.safe_sparse_dot(transformed, pinversed_components.T)
        loss = ((dataset.data - reconstructed)**2).mean()
        np.append(losses, loss)
        losses = norm = np.linalg.norm(dimreducer.components_, ord='fro')
        norms = np.append(norms, norm)
      lossmeans = np.append(lossmeans, np.mean(losses))
      normmeans = np.append(normmeans, np.mean(norms))
      lossvars = np.append(lossvars, np.var(losses))
      normvars = np.append(normvars, np.var(norms))

    # plt.plot(cs, normmeans, label='Mean of RP mixing matrix frobenius norm')
    # plt.plot(cs, lossmeans, label='Mean of RP Reconstruction loss')
    plt.plot(cs, normvars, label='Variance of RP mixing matrix frobenius norm')
    plt.plot(cs, lossvars, label='Variance of RP Reconstruction loss')
    plt.xlabel(f'number of components')
    plt.legend()
    plt.title(f'RP variances on {reruns} reruns\non {dataset.name} Dataset')
    plt.show()
      
  # TODO: random runs lineplot x=components y=scalar color=[variance(loss), variance(forbenius norm)]
  # Fixed reruns=10


def nmf(dataset, components, main=True,
        use_components_percent=False, vis_components=False,
       return_dimreduced=False, return_loss=False):
  if main:
    if use_components_percent:
      _components = components
      components = int(dataset.F * components)
      # print(f"{_components} of components are {components} components")

    dimreducer = NMF(n_components=components, solver='mu', max_iter=5000)
    dimreducer.fit(dataset.data, y=dataset.target)
    transformed = dimreducer.transform(dataset.data)
    reconstructed = dimreducer.inverse_transform(transformed)
    loss = ((dataset.data - reconstructed)**2).mean()

    if return_dimreduced:
      return transformed
    
    if return_loss:
      return loss
    
    # print(f"Reconstruction loss: {loss}")

    if vis_components:
      print(f"{_components} of components are {components} components")
      Plots.vis_components(components, dataset, dimreducer)
        
  # TODO: Vis components

  
def dr(dataset, comp_dr=False, to_nn=False, cs=[], use_components_percent=True):
  
  if comp_dr:
    # Lineplot color=[pca, rp, nmf] x=components y=loss
    plt.plot(cs, [pca(dataset, c, use_components_percent=use_components_percent, return_loss=True) for c in cs], label='PCA')
    plt.plot(cs, [rp(dataset, c, use_components_percent=use_components_percent, return_loss=True) for c in cs], label='RP')
    plt.plot(cs, [nmf(dataset, c, use_components_percent=use_components_percent, return_loss=True) for c in cs], label='NMF')
    plt.legend()
    plt.title(f'Reconstruction loss of Dimensionality Reduction algorithms\non {dataset.name} Dataset')
    plt.xlabel('Fraction of components')
    plt.ylabel('Reconstruction loss')
    plt.show()
  
  if to_nn:
    # Use pca ica rp nmf return_dimreduced=True
    # Use train_test_split
    # Lineplot color=drs x=components y=accuracy
    # Compare with Assignment 1's 98%
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, shuffle=True)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    train_dataset = Dataset(X_train, y_train, dataset.C, dataset.F, dataset.N * 0.7, dataset.target_names)
    test_dataset = Dataset(X_test, y_test, dataset.C, dataset.F, dataset.N * 0.3, dataset.target_names)
    dimreduced_train_drs = [
      [pca(train_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [ica(train_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [rp(train_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [nmf(train_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs]
      #,
      #[nmf(train_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs]
    ]
    dimreduced_test_drs = [
      [pca(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [ica(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [rp(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs],
      [nmf(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs]
      #,
      #[nmf(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True) for c in cs]
    ]
    accs_drs = [
      [],
      [],
      [],
      []
    ]
    dr_names = ['PCA', 'ICA', 'RP', 'NMF']
    
    for idx_dr, (dimreduced_train_components, dimreduced_test_components) in enumerate(zip(dimreduced_train_drs, dimreduced_test_drs)):
      # print(f'Processing {dr_names[idx_dr]} with dimreduced_train_components {len(dimreduced_train_components)} and dimreduced_test_components {len(dimreduced_test_components)}')
      for idx_component in range(len(dimreduced_train_components)):
        # np.full((5, ), 100
        nn = MLPClassifier(hidden_layer_sizes=[100, 100, 100, 100, 100], learning_rate_init=0.001, max_iter=5000,
                          solver='lbfgs', activation='relu')
        # scaler = StandardScaler()
        # scaler.fit(dimreduced_train_components[idx_component])
        # nn.fit(scaler.transform(dimreduced_train_components[idx_component]), train_dataset.target)
        nn.fit(dimreduced_train_components[idx_component], train_dataset.target)
        
        # acc = nn.score(scaler.transform(dimreduced_test_components[idx_component]), test_dataset.target)
        acc = nn.score(dimreduced_test_components[idx_component], test_dataset.target)
        accs_drs[idx_dr].append(acc)
      # print(accs_drs[idx_dr])
        
    for i in range(len(accs_drs)):
      plt.plot(cs, accs_drs[i], label=dr_names[i])
    plt.legend()
    plt.title(f'Performance of Neural Network\non Dimensionality Reduced training data\non {dataset.name} Dataset')
    plt.xlabel('Fraction of components')
    plt.ylabel('Test-time accuracy')
    plt.show()
      


def cluster_dr(dataset, components, vis_cluster=False, to_nn=False, cs=[], use_components_percent=True, backup=None):
  # First, fixed each of best components
  if vis_cluster:
    # Use for dataset
    #         for pca ica rp nmf return_dimreduced=True
    #             for kmeans em
    # Subplot clusters
    # Compare vis with Part 1's Clustering
    # Use for pca ica rp nmf
    #         by vis
    #         by higher v-measure
    
    # TODO: KMeans 9 classes only?
    pca_dataset = Dataset(
      pca(dataset, components, use_components_percent=use_components_percent, return_dimreduced=True),
      dataset.target, dataset.C, dataset.F, dataset.N, dataset.name, dataset.target_names
    )
    acc1k, v1k = kmeans(pca_dataset, vis=True, return_acc_v=True, method_desc='PCA-dimensionality-reduced-KMeans-clustered')
    acc1e, v1e = em(pca_dataset, vis=True, return_acc_v=True, method_desc='PCA-dimensionality-reduced-EM-clustered')
    
    ica_dataset = Dataset(
      ica(dataset, components, use_components_percent=use_components_percent, return_dimreduced=True),
      dataset.target, dataset.C, dataset.F, dataset.N, dataset.name, dataset.target_names
    )
    acc2k, v2k = kmeans(ica_dataset, vis=True, return_acc_v=True, method_desc='ICA-dimensionality-reduced-KMeans-clustered')
    acc2e, v2e = em(ica_dataset, vis=True, return_acc_v=True, method_desc='ICA-dimensionality-reduced-EM-clustered')
    
    rp_dataset = Dataset(
      rp(dataset, components, use_components_percent=use_components_percent, return_dimreduced=True),
      dataset.target, dataset.C, dataset.F, dataset.N, dataset.name, dataset.target_names
    )
    acc3k, v3k = kmeans(rp_dataset, vis=True, return_acc_v=True, method_desc='RP-dimensionality-reduced-KMeans-clustered')
    acc3e, v3e = em(rp_dataset, vis=True, return_acc_v=True, method_desc='RP-dimensionality-reduced-EM-clustered')
    
    nmf_dataset = Dataset(
      nmf(dataset, components, use_components_percent=use_components_percent, return_dimreduced=True),
      dataset.target, dataset.C, dataset.F, dataset.N, dataset.name, dataset.target_names
    )
    acc4k, v4k = kmeans(nmf_dataset, vis=True, return_acc_v=True, method_desc='NMF-dimensionality-reduced-KMeans-clustered')
    acc4e, v4e = em(nmf_dataset, vis=True, return_acc_v=True, method_desc='NMF-dimensionality-reduced-EM-clustered')
    
    methods = ['PCA', 'ICA', 'RP', 'NMF']
    plt.plot(methods, [acc1k, acc2k, acc3k, acc4k], 'g-', label='KMeans accuracy')
    plt.plot(methods, [v1k, v2k, v3k, v4k], 'g--', label='KMeans v-measure-score')
    plt.plot(methods, [acc1e, acc2e, acc3e, acc4e], 'b-', label='EM accuracy')
    plt.plot(methods, [v1e, v2e, v3e, v4e], 'b--', label='EM v-measure-score')
    plt.legend()
    plt.xlabel('Dimensionality Reduction algorithm')
    plt.title(f'Performance Comparison of Dimensionality Reduction algorithms\n coupled with Clustering algorithms\non {dataset.name} Dataset\non{components} of components')
    plt.show()
  
  if to_nn:
    # Use for pca ica rp nmf return_dimreduced=True
    #         for kmeans em return_aug_data=True
    # Lineplot color=drs x=components y=accuracy solid_dot=clusters
    # Compare with Assignment 1's 98%
    
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, shuffle=True)
    train_dataset = Dataset(X_train, y_train, dataset.C, dataset.F, dataset.N, dataset.target_names)
    test_dataset = Dataset(X_test, y_test, dataset.C, dataset.F, dataset.N, dataset.target_names)
    
    #x = pca(test_dataset, 0.1, use_components_percent=use_components_percent, return_dimreduced=True)
    #y = np.ones(int(dataset.N * 0.3)+1)[:, np.newaxis]
    #print("x", np.array(x).shape)
    #print("y", np.array(y).shape)
    #z = np.hstack([x, y])
    #print("z", np.array(z).shape)
    
    
    dimreduced_train_datasets = [
      [Dataset(pca(train_dataset, c, use_components_percent=True, return_dimreduced=True), train_dataset.target, 
               dataset.C, dataset.F, int(dataset.N * 0.7), dataset.name, dataset.target_names) for c in cs],
      [Dataset(ica(train_dataset, c, use_components_percent=True, return_dimreduced=True), train_dataset.target, 
               dataset.C, dataset.F, int(dataset.N * 0.7), dataset.name, dataset.target_names) for c in cs],
      [Dataset(rp(train_dataset, c, use_components_percent=True, return_dimreduced=True), train_dataset.target, 
               dataset.C, dataset.F, int(dataset.N * 0.7), dataset.name, dataset.target_names) for c in cs],
      [Dataset(nmf(train_dataset, c, use_components_percent=True, return_dimreduced=True), train_dataset.target, 
               dataset.C, dataset.F, int(dataset.N * 0.7), dataset.name, dataset.target_names) for c in cs]
    ]
    
    aug_train_datasets = [
      (kmeans(_train_dataset, return_aug_data=True), em(_train_dataset, return_aug_data=True)) for train_datasets_per_dr in dimreduced_train_datasets for _train_dataset in train_datasets_per_dr
    ]
    
    
    dimreduced_test_drs = [
      [np.hstack([pca(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True), 
                  np.ones(int(dataset.N * 0.3)+1)[:, np.newaxis]]) for c in cs],
      [np.hstack([ica(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True),
                  np.ones(int(dataset.N * 0.3)+1)[:, np.newaxis]]) for c in cs],
      [np.hstack([rp(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True),
                  np.ones(int(dataset.N * 0.3)+1)[:, np.newaxis]]) for c in cs],
      [np.hstack([nmf(test_dataset, c, use_components_percent=use_components_percent, return_dimreduced=True),
                  np.ones(int(dataset.N * 0.3)+1)[:, np.newaxis]]) for c in cs]
    ]
    
    accs_drs = [
      ([], []),
      ([], []),
      ([], []),
      ([], [])
    ]
    dr_names = [
      ('PCA-KMeans', 'PCA-EM'), 
      ('ICA-KMeans', 'ICA-EM'),
      ('RP-KMeans', 'RP-EM'),
      ('NMF-KMeans', 'NMF-EM')
    ]
    fmts = [
      ('b-', 'b--'),
      ('g-', 'g--'),
      ('r-', 'r--'),
      ('c-', 'c--')
    ]
    
    for idx_dr, (aug_train_dataset, dimreduced_test_components) in enumerate(zip(aug_train_datasets, dimreduced_test_drs)):
      for idx_clusterer in range(2):
        for idx_component in range(len(aug_train_dataset[idx_clusterer])):
          # np.full((5, ), 100
          nn = MLPClassifier(hidden_layer_sizes=[100, 100, 100, 100, 100], learning_rate_init=0.001, max_iter=5000,
                            solver='lbfgs', activation='relu')
          # scaler = StandardScaler()
          # scaler.fit(dimreduced_train_components[idx_component])
          # nn.fit(scaler.transform(dimreduced_train_components[idx_component]), train_dataset.target)
          nn.fit(aug_train_dataset[idx_clusterer][idx_component], train_dataset.target)

          # acc = nn.score(scaler.transform(dimreduced_test_components[idx_component]), test_dataset.target)
          acc = nn.score(dimreduced_test_components[idx_component], test_dataset.target)
          accs_drs[idx_clusterer][idx_dr].append(acc)
        # print(accs_drs[idx_dr])
        
    for idx_dr in range(len(accs_drs)):
      for idx_clusterer in range(2):
        plt.plot(cs, accs_drs[idx_dr][idx_clusterer], fmts[idx_dr][idx_clusterer], label=dr_names[idx_dr][idx_clusterer])
    plt.legend()
    plt.title(f'Performance of Neural Network\non Dimensionality Reduced training data augmented with clustering feature\non {dataset.name} Dataset')
    plt.xlabel('Fraction of components')
    plt.ylabel('Test-time accuracy')
    plt.show()

