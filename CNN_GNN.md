Contents lists available at ScienceDirect

Biomedical Signal Processing and Control

journal

www_elsevier.

## Data availability

Links listed below are the data deposition URLS from Data availability section. Please verify the links are valid. This page will not

or print. They are displayed in the proof pdf for review purpose only.

in the article PDF file

appear

https://www.Kaggle. retinal-oct-images-splitted/data

izi i/retinal-oct-images-splitted/data Dataset Link: https://www_kaggle.com/d

Dataset Link:


Contents lists available at ScienceDirect

## Biomedical Signal Processing and Control

journal

www.elsevier.

## A hybrid CNN-GNN approach for multi-class classification and segmentation of retinal OCT images

Aryaman Chandra Aniket Choudhury 2, Prateek Kavya Singh Reenu Rani

- School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, 632014, India

- ® Department of Biosciences and Bioengineering, Indian Institute of Technology Roorkee, 247667, Uttarakhand, India

- Department of Mathematics, School of Advanced Sciences, Vellore Institute of Te , Vellore, 632014, India

## ARTICLE INFO

Dataset link: hitps://

i

a,

Deep Optical learning coherence tomography (OCT) Hybrid CNN-GNN model

Retinal se

image

Convolutional neural networks (CNN)

## ABSTRACT

Optical Coherence

of the retina

disorders

Although deep learning methods and Convolutional Neural Networks (CNN) offer impressive accuracy for segmentation, anomaly detection, and classification, they fail to capture the holistic structural relationships

spanning

the

segmentation

classification,

a

for structural reasoning. Retinal features and lesions are represented as nodes of the graph, ions. allowing

retain spatial

to encode anatomical

framework anomaly detection, disease classification, and reliability as compared to baselines using improves

CNN only and outperforms a fully supervised segmentation baseline on the

The

88.7% vs 82.4%) without requiring any pixel-level

Grad-CAM and GNN attention to visualize clinically relevant insights into regions

and structural markers of interest such as layer thinning and connectivity disruptions. Experiments were conducted on the publicly available Mendeley OCT dataset, with additional evaluation on the OCTSk dataset for supervised comparison.

frameworks suggest the necessity of bringing together different tools to enhance the generality of retinal pathology diagnosis systems [1].

Optical coherence tomography (OCT) [2] has emerged as an in- dispensable imaging tool in current-day ophthalmology because of its ability to provide high resolution, non-invasive cross-sectional imag- ing of retinal anatomy. It enables clinicians to view layer anatomy, fluid spaces, and structural pathology consistent with DME and CNV and drusen. Despite these advantages, interpretation of OCT volumes through manual inspection is a tedious process and can be highly variable between researchers and not easily scalable for population screening or high throughput clinical setting. These challenges call for automated methods that can expedite the process and enable stan- dardization of OCT image interpretation, while still providing clinically interpretable information. Deep learning and convolutional neural net- works (CNNs) [3-7], have made great strides in the realm of automated OCT image analysis because of their ability to learn discriminative

Retinal disorders are a leading cause of reversible vision impairment globally. Diabetic Macular Edema (DME), choroidal neovascularization (CNV), and drusen are common diseases that can cause significant and usually irreversible loss of central vision if missed early. Early detection and treatment are critical: treatment with anti-VEGF injections for CNV and DME is most effective when treated early, and early detection of drusen can be used in conjunction with ongoing retinal degenerative disease monitoring.

Combining multiple OCT and fundus imaging streams has been confirmed to improve the specificity and sensitivity of the automatic classification of retinal diseases in the recent literature. The retinal disease diagnosis frameworks that utilize multiple streams confirm that combining structural and textural information leads to an increase in the number of distinguishing features of retinal pathologies. These

(R. Rani).

https://doi.org/10.1016/j.bspc.2026.111299

Received 11 November 2025; Received in revised form 27 May 2026; Accepted 11 August 2026

www kaggle.com/datasels litted/d;

Tomography

diagnosis

(OCT)

is a crucial

component of

the assessment

and

of

several

(CNV)

(DME).

such as Drusen,

Choroidal

Neovascularization

and diabetic

macular

edema

layers. Therefore,

a hybrid

retinal

framework

is introduced

based

for automatic

on CNNs

and Graph

U-Net architecture,

(GNNs) models

decoder based

on the

Neural

Networks

The multi-stage

and refine

(Dice:

OCTSk

dataset

likely i

through lesions,

have

to

## 1. Introduction


features from images directly for classification and segmentation. CNNs are good at finding local texture and edge information; however, due to their inherent local nature of receptive field, they might lack in capturing large scale anatomical context and relationship across retinal layers which are generally very crucial for diagnosing. This problem is addressed by using U-net like architecture consisting of encoder decoder type network [8,9]. However, even outputs from the segmen- tation task might fall short in terms of capturing the spatial dependency and anatomical relationship for multi-class diagnosis.

Alongside classical CNNs architectures, recent architectures such as DenseNet and EfficientNet have also been widely researched for

the of medical image analysis. DenseNet architecture, intro- purpose duced by Huang et al. [10], uses layer-to-layer connectivity to promote reusing of features and efficient gradient propagation, allowing extract- ing fine-structural features. EfficientNetarchitecture, proposed by Tan and Le [11], uses compound scaling method taking into account depth, width and resolution of a network, reaching state-of-the-art results while being computationally efficient. Although these architectures have shown their superiority in standard image classification problems, their performance in hybrid approaches that use graph reasoning re- mains unclear especially regarding interaction between their feature representation and relational modeling in medical imagery.

Nonetheless, despite many advancements made, current approaches possess multiple limitations. CNN architectures mainly focus on local

spatial features and cannot account for long:

ical depen-

dencies across retinal layers [12,13]. Although transformer architec- tures allow for better global modeling, they are computationally expen- sive and require large annotated dataset [14,15]. Moreover, almost all segmentation-based approaches require dense pixel-level annotations which are costly and inconsistent in clinical practice [16,17].

To solve these this presents a CNN-GNN hybrid ap-

issues, paper

proach which models the anatomical relationships between different parts using graphs but still retains the feature extraction efficiency of the CNNs [18]. The novel contribution comes from the use of

localization along ith i the graph-based reasoning

weakly allowing

supervised for the precise

even without dense annotation data.

In order to solve the aforementioned problems, a hybrid system architecture including three mutually complementary components is suggested. First is a ResNet based CNN backbone for hierarchical feature extraction, next is a U-Net like decoder that generates detailed segmentation masks highlighting the lesion and layer areas, and third are Graph Neural Networks (GNNs) which explicitly incorporate spatial

and relational properties i of retinal ility subregions.

[19] ing

Grad-CAM for

method lesion-

approach

is

as an

focused heatmaps. In the suggested architecture, the CNN encoder generates node features for anatomically significant areas (generated either from the segmentation of the U-Net or from a patch/superpixel representation) and the GNN works on a graph whose nodes are the

regional

ions and edges are

ical adjacency or simi-

larity. The architecture proposed here combines the strength of U-Net localization capability, representational power of CNNs, and the re- lational reasoning capacity of GNNs to capture both local and global

retinal

structure.

the performance difference between different

To measure

to modeling graphs, the hybrid pipeline is benchmarked against three baselines that use the same ResNet model to extract features but differ in their graph operator: ResNet + GATv2, ResNet + GIN, and ResNet + GraphSAGE. The choice of baselines was made to investigate different properties of graph reasoning separately. GATv2 (Graph Attention

Network v2) [20], for example, learns adaptive i

weights

per edge using attention mechanism, making it possible to put more

weight on the edges that are important from a clinical

GIN

(Graph Isomorphism Network), for its part, has very high theoretical expressive power when it comes to distinguishing between graph structures. GraphSAGE, finally, implements an inductive approach to

building a nei

ing graph

and is thus

well-suited for generalization to new patients or images. To distinguish the impact of graph structure on classification robustness, consistency, and generalization, the encoder is fixed and the design of the graph module is varied.

Standard classification and segmentation metrics (accuracy, AUC, sensitivity, specificity, Dice, IoU) are used to evaluate the performance of the model. Grad-CAM is used to visualize lesion localization on multiple models and rank their interpretability. The experiments are designed to demonstrate that CNN-based feature extraction coupled with U-Net style segmentation and graph-based relational reasoning achieves more clinically interpretable OCT classification than the use of local convolutional context in isolation.

## 2. Related work

The integration of deep learning strategies such as Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) has sig- nificantly advanced the diagnosis and classification of ocular diseases.

‘This section will focus on the latest advancements ity

detection,

in the areas of

of retinal OCT images where there have been advances from simple local features to relation-based models [1,2].

et al. (2024) carried out a comparative study based on ViT and found that transformer-based architectures are effi-

cient for capturing global contextual dependencies through attention mechanisms for retinal OCT classification [3]. Barbosa et al. (2025) proposed a modified VGG-19 architecture for retinal OCT images classi- fication that proved to be efficient in terms of using deep convolutional networks for discrimination and classification of features in multiple retinal disease classes [4]. Jaimes et al. (2025) examined transfer learning approach to classify OCT images using VGG-16 model, proving that a pre-trained model can be effective in diagnoses even when the amount of imaging data is limited [5]. Another example could be given

by and

et al.

VGG, and

in terms of high frequency feature extraction for retinal damage detec- tion [6]. Moreover, Karri et al. (2017) investigated transfer learning, revealing that pretraining on a vast number of nonmedical images greatly improves performance in a data constrained task in medical field [7]. Additionally, ReLayNet, a fully convolutional network for layer and fluid boundaries [8], helped improve segmentation accuracy, whereas Karn et al. (2023) used attention-based U-Nets to increase

fluid segmentation

subretinal

After the

searchers paid attention to optimizing the data flow inside the network. In 2017, Huang et al. came up with Densely Connected Convolutional Network (DenseNet), where each network layer is directly connected to all other layers to maximize the reuse of features [10]. In 2019, Tan and Le suggested EfficientNet that uses compound scaling coefficient to scale network depth, width and resolution in an efficient way [11]. Mienye and Viriri (2025) stated that despite the excellent performance on grid-like data, GNNs become a necessity in the irregular medical imaging topologies [12]. The in modern CNNs to research model the includes dependence the integration between

[13] and use of ViT to detect vitelliform lesions with high

sensitivity [14,15]. Moreover, weakly supervised learning systems that were developed by Mehta et al. and Liu et al. proved that models can localize biomarkers based on image-level labels without complex

pixcllevel labeling | [16,171

lar recently. Specifically, Qian et al. (2025) proposed a Pyramid Vision

Graph C

OCT features [18], whereas Selvaraju et al. introduced Grad-CAM to make sure that these sophisticated models stay interpretable to the doctors in heatmaps [19]. Brody et al.’s theoretical advancements in

have been made

that

and

Sathyan

compared

including

(2023),

models,

who

CNN

ResNet

deeper

showed

how

effective

neural

networks

more

are

precision

[9].

was leted, the re-

of CNN

archi

reasoning has become increasingly popu-

Network (PV-GCN), reasoning over multi-scale


Graph Attention Networks (GATS) have also helped overcome problems related to static attention, making GATs more applicable to medical

- (i) Labeling burden: Multiple high-performing approaches rely on either dense pixel-wise masks (not commonly available in cataract surgery) or larger, curated labels that are costly to develop and often inconsistent across centers.

it was recently

data [20].

that GNNs can

i

be more efficient than Transformers in segmentation tasks due to better representation of geometric complexity of biological structures by Xiao et al. (2025) [21]. This advancement is reflected in the results of hybrid CNN-GNN models in brain tumor and pancreatic cancer classification [22,23], as well as the identification of particular retinal ‘biomarkers in ensemble deep learning models [24].

- (ii) Limited structural reasoning: Purely convolutional pipelines place emphasis on local texture and frequently underutilize the layered topology of the retina, despite its importance in distinguishing between CNV, DME, and drusen.

- (iii) Generalization and interpretability: Supervised models can over- fit to the acquisition protocol under which they are trained, and often the saliency of the model is not grounded in structural rationale when used for clinical audit.

Literature reviews carried out by Hassan et al. (2024) and the framework of HyReti-Net introduced by Yang et al. (2025) highlight

the

and

disease diagnosis [25,

of

26]. Precision in medical applications can be achieved by means of Random Forest algorithms for layer thickness measurement and au-

Recent work tends to address these issues with graphs or weak/self

supervision; however,

there has not been a rigorous study that seeks to

self- macular

segmentation deep models to

tomatic

edema and

[27,28].

in

the area

Progress

develop

a pipeline

offering

labeling efficiency,

a prior

structure-aware

learning

of vided an

has

opportunity

pro- and

reasoning

connectivity

of the retina,

for anatomical

and

usable lesion

for

utilize unlabelled

fundus

localizations.

OCT images, thus providing improved generalization among various

‘To address these we adopt an unsupervised/weakly-supervised gaps, hybrid approach. In summary, a CNN encoder provides local features, a GNN module enforces anatomical consistency over a patch graph, and lightweight localization heads directly output class-specific heatmaps without the need for pixel-wise annotations or intricate labeling. This design enables labeling efficiency, structure-aware reasoning, and clin- ically interpretable outputs that are more robust across datasets and devices than traditional fully supervised approaches.

groups patients [29]. Clinically applicable deep learning solutions

of

provide highly efficient diagnostic and referral functions using OCT images [30]. Following advances in self-supervised learning provide

further

learnin;

of

that en-

hance the capability of disease detection [31-33]. Layer segmentation has been taken to a completely new level of automation with the help of graph search hybrids and full automation of quantification [34-37]. These results were achieved due to advances in deep residual learn- ing [38] and topological mapping approaches to studying biological tissues [39-41].

## 4. Methodology

Robustness is ensured using state-of-the-art data augmentation tech- niques that include using OCT patches synthesized with help of GANs and total variation denoising in order to cope with signal to noise ratio problems [42-46]. Model validation is usually done using big scale datasets that can be found in Kermany and Arikan datasets

## 4.1. Datasets

The dataset used for this study was the Retinal Optical Coherence

kaggle. Tomography | (OCT) dataset from Mendeley OCT

disease:

The dataset consists of four classes of the

vascularization (CNV), Diabetic Macular Edema (DME), Drusen, and Normal. In addition, for supervised model comparison, the OCTSk

dataset [48] dataset. provides labeled OCT

for

disease classification. The

validation (10%-15%), and test (10%-15%) sets. All OCT image data were converted to grayscale. The data were preprocessed with standard pipelines that consisted of resizing to 256 x 256 px, normalized with mean x = 0.5 and standard deviation ¢ = 0.5, and data augmen- tation training sets. Data augmentation consisted of random in the horizontal flips (p = 0.5), image rotation through +10°, and brightness

steps supported

and contrast adj of +20%. The

robustness generalization while preserving important diagnostic and features.

[47]

litted/data).

allowing multiple-graded

ions testing [47,48]. Rei

Choroidal

Neo-

-

based transfer learning and local contrast learning have recently been used for improving the process of adaptation of models to clinical setting [49,50]. Importantly, Attention-based Deep Multiple Instance Learning (MIL) proposed by et al. [51]and region hiding tech- niques like Hide-and-Seek [52] have shown that occlusion of dominant regions leads to the discovery of more subtle but semantically impor- tant features. Motivated by this finding, our “erase trick” forces the model to explore less obvious pathological evidence. Prototypical few- shot segmentation and various Swin Transformers together with the attention-based MIL and region-hiding methods are the current state-of- the-art in efficient retinal image processing [53,54]. Moreover, hybrid

was utilized. This

images

is commonly

and

used

as a

learning

in retinal

dataset

ised

split

training (70%-80%),

into

was

networks that integrate

ional networks and

net-

ing

works have been efficiently used in the field of retinal image analysis. For instance, ADCFormer [55] employs adaptive depth-wise convo- lution together with transformer networks for the task of macular edema segmentation in OCT images where the feature representation becomes more adaptable. On the contrary, SAFP-Net [56] uses spatially augmented feature pyramid for microaneurysm detection in retinal images.

## 4.2. i i and

‘The experiments took place in a deep learning environment to max- imize both reproducibility and computational efficiency. The training process was done using PyTorch (v2.4.0) with its extension, PyTorch Lightning, to provide modularity, logging, and checkpoints. A worksta- tion with an NVIDIA RTX 3090 (24 GB VRAM), Intel Xeon Processor, and 128 GB RAM was utilized. The operating system was Ubuntu 20.04 LTS with the appropriate CUDA for GPU acceleration. Tensor- Board was used as part of the experimental setup for plotting training

heatmaps. TensorBoard was

attention and localization

maps,

curves,

also used within the context of

‘The training set included early stopping callbacks and model check- pointing, which would automatically stop the training when conver- gence was detected, while keeping the best weights checked.

Taken together, these studies demonstrate that while CNNs are effective at local feature extraction, they often struggle with global structural relationships. This motivates our proposed integration of GNN-based relational reasoning and attention-guided erasing, aiming

feature extraction classi and fine-grained

to bridge the

high-level

gap between

structural

in retinal OCT

i

ility

## 3. Motivation

Although CNN- and transformer-based OCT approaches have pro- duced strong results, there are still three practical gaps to be addressed:

Lightning loggers for managing log files.


*Fig. 1. Overall architecture of the proposed hybrid CNN-GNN model. The network integrates CNN-based local feature extraction and GNN-based spatial reasoning*

*modules to capture both fine and contextual information.*

Mixed-precision (16-bit) training was implemented to promote com- putational efficiency and decrease memory requirements without loss of numeric precision. During training, an Exponential Moving Average (EMA) method with a decay factor of 0.999 to increase robustness is used. This method ensured that parameters were updated via smooth rolling averages with time, again leading to enhanced generalization during inference.

Fig. 1 illustrates a hybrid CNN-GNN framework as implemented in our research. A single-channel OCT B-scan is encoded by a ResNet-34 backbone (grayscale first conv), from which the layer-3 feature map is taken (16 x 16) and the lower-resolution x2 map is saved for a later skip. The 16 x 16 tensor is flattened into patch tokens and is then used to construct a graph of a 4-connected grid graph with patch embeddings and relative position offsets as the nodes. Then, messages

are passed through two layers of multi-head GATv2, and node outputs are projected and pooled globally as output logits. A class-specific Multiple Instance Learning (MIL) head will output image-level logits, including with an erased-MIL branch to mask top-attention patches to reduce shortcut reliance. In parallel, a lightweight localization head using the x3 feature map provides a per-pixel prior that is upsampled and injected into a U-Net-style decoder together with (x2,x3) to yield logis for segmentation and heatmaps for class-specific attention at the pixel level.

Fig. 2 illustrates the end-to-end pipeline executed in our environ- ment. OCT images are obtained, preprocessed to form 256 x 256 images, and augmentation (flips, small rotations, brightness/contrast jitter, noise, and random erasing) is applied. Batches of images are loaded and passed through the CNN encoder to get feature maps,


*Fig. 2. Overall flow diagram of the proposed deep learning pipeline illus- trating the sequential stages of data preprocessing, feature extraction, model training, and evaluation.*

the tokenized 16 x 16 feature map is then used to create a grid

graph for processing GATv2 to produce spatially informative features used in the MIL classification head with the erase trick. In parallel,

decoder takes

of encoder features and

weak

fon

a

the upsampled prior to create dense activation maps. Training happens with PyTorch + Lightning with AMP (16-bit), EMA (a = 0.999), early

stopping,

ion, image-level

inting, with

for

predictions, localization heatmaps, and weak segmentation outputs for visualization and analysis are exported.

## 4.3. Image augmentation and preprocessing

Prior to entering the training pipeline, Optical Coherence Tomog- raphy (OCT) images were subjected to a standard process to retain consistency for each image, improving model efficiency. Each OCT image was scaled down to 256 x 256 px to standardize input size [50].

*Fig. 3. ResNet-34 backbone architecture diagram showing the stacked residual blocks that extract discriminative spatial features from OCT images [38].*

To keep localized features while preserving spatial feature organization, each scaled image was divided into a non-overlapping grid of 16 x 16 for a total of 256 patches per image to be input into CNN-GNN as atomic elements. Several augmentation processes were performed to

generalization

capabilities

overfitting.

increase processes

and

minimize

These [42],

included randomly

flipping

(horizontally

vertically)

and

applying minor rotations of +10°, adjusting random brightness and contrast, random Gaussian noise injections, and random erasing to simulate occlusions. To guarantee that every epoch experienced new variants of the input distribution, augmentation methods were done online, in real-time, using the Albumentations library during training.

## 4.4. Convolutional Neural Network (CNN) backbone

The CNN encoder used ResNet-34 [38], as illustrated in Fig. 3, which was modified for single-channel OCT images by replacing the first convolutional layer to accept grayscaled inputs. This architec- ture included four sequential processing stages (layerl-layerd), which abstracted image features and had the following behavior: stage 1 focused on physical edge and texture characteristics, and stages 2 and 3 learned intermediate representations related to retinal structures.


Stages 1, 2, and 3 focused on lower categories of representations, while stage 4 learned a high level semantic feature representation that was invariant to small shifts and noise in the image. Additionally, low and intermediate layers were retained and generated outputs that allowed for both fine- and coarse-grained feature maps to be passed downstream into the GNN and definitions of the decoder modules. The high-level feature maps output by ResNet-34 were reshaped into patch-level embeddings [12,18], which represented the first graph node representations in the next GNN module.

## 4.5. Graph Neural Network (GNN) module

doing CNN-based

extraction,

After

interpreted

as a node

was

between patch nodes based on spatial adjacency using a 4-connected neighborhood in the 2D grid [40]. This approach maintained the intrinsic topological properties of OCT images of the retina and facili-

tated relation-based inference the patches. To examine various

among

approaches of graph learning, three GNN approaches were considered. The first approach was a graph attention network (GATv2) [20], which used two stacked layers, each having four attention heads via atten- tion mechanism based on features similarity differentiated by relative positional encoding). This helped the network to focus on informative neighboring patches in an adaptive manner. Another graph learning technique used was the GraphSAGE for inductive neighborhood aggre-

gation graph learning technique. Next, a Graph

(GIN) model was tested that similarly relied on the concept of induc- tive neighbor aggregation; however, it was built based on using the multilayer perceptron for aggregating features of neighbors to obtain theoretically proven structural expressiveness. The analysis showed that the GATv2 was able to provide the optimal balance between local relations and global context as the graph learning model. While the graph itself is built on the basis of the fixed 4-connected adjacency

in the proposed GNN architecture is fundamentally different space, from traditional CNN architecture due to the usage of adaptive atten- tion. Thus, the Graph Attention Network v2 (GATv2) uses learnable importance weights to give the data-dependent importance to each neighboring node during message passing. This allows giving priority to the more important regions of the local neighborhoods. Mean- while, standard CNN use convolutional layers with fixed convolutional kernels that have the same weight across space, thus being unable to perform any adaptive modulation of relations between neighbor- ing regions. Moreover, this graph-based approach provides a flexible framework that can be generalized to include non-local or feature- wise connections in future research, which is difficult in standard CNN architectures [12,20].

patch embedding

feature

each

in a graph

edges

[18], with

ism Network

## 4.6. Classification with the erase trick

The fused CNN-GNN embeddings [18] were fed into a Multiple

head [511 to predict

Instance Learning (MIL)-based

classification

of four diagnostic

ies: Choroidal

abetic Macular Edema (DME), Drusen, or Normal. To reduce over- reliance on the most discriminative regions, an erase trick was applied during training, in which patches with the highest attention scores were selectively masked. This strategy encouraged the model to discover ad- ditional disease-relevant evidence across the image, thereby enhancing classification and improving interpretability by preventing decisions from being dependent on a limited set of patches. Let A € RY represent the attention scores over N patches. The top-k patches are selected and masked:

The modified feature is: map

one Di-

(CNV),

Neo

‘The masked attention is then upsampled using bilinear inter- map polation to generate a dense prior map P, which is incorporated into the decoder as spatial guidance [52].

## 4.7. Segmentation with decoder

Besides classification, the model also included two weakly super- vised localization heads to develop interpretable spatial outputs. The

performed

first was a CNN-based decoder,

module,

tures into pixel-level activation maps. The second was a GNN-based decoder, which utilized relational reasoning across the different patches to highlight the global disease-relevant sections of the image. The GNN decoder complemented the CNN decoder, which was more local in nature. The combination of CNN and GNN decoders produced class- specific heatmaps to help the clinician direct their attention towards the pathological areas in the absence of pixel-wise annotations. The spatial localization maps were also regularized over the total variation and entropy during training by conditioning on a A weighted parameters, which encouraged spatial coherence and utilization of diverse features. The decoders functioned to produce outputs which were similar to

via weakly sup means, providing a bridge be-

tween classification accuracy and visual interpretability for retinal OCT analysis.

as a lightweight

which

the patch-level

discriminative

fea-

ing

## 4.8. Comparative study of GNN variants

To investigate the impact of varying GNN architectures in a hybrid CNN-GNN framework, a comparison experiment of three variations: Graph Attention Networks (GATv2), GraphSAGE, and Graph Isomor- phism Networks (GIN) is implemented. Each model was implemented with a ResNet-34 backbone, and the training paradigm, preprocessing, and all optimization remained constant across models in order to properly investigate performance.

The GATv2 variant consisted of two stacked attention layers with four attention heads within each layer. The GATv2 utilized similarity of features as well as relative positional encodings to modulate the amount

given to each neighboring patch, thus allowing the model

i

of

to place an emphasis on patches of relevance to disease while ignoring unimportant contextual information. GraphSAGE, on the other hand, employed an inductive sampling strategy across neighboring patches, where the node representation is updated through information aggrega- tion with just random neighboring node(s) to introduce a more efficient computing cost and scalability for the GNN in larger graphs that are still able to retain the comparable local structure information. Lastly,

a multilayer

utilized

GIN

perceptron

and provides

better

(i.e., distinguishes) isomorphisms when compared to the more standard message-passing networks.

The models were evaluated on a combination of the classification and weak localization tasks. GATv2 always performed better than the other two models and was able to show better sensitivity and specificity. The attention mechanism used in GATv2 allowed for a more precise modeling of diseased regions. GraphSAGE performed acceptably and was able to do so more efficiently, this was helpful in situa- tions with differentials in resources. Finally, GIN was mathematically expressive but did not achieve classify OCTs accurately, particularly due to its MLP-aggregation overfitting without a level of class uncer- tainty. Overall, GATv2 demonstrated the greatest effectiveness for the identified framework for OCT analysis, with robust classification and interpretations.

aggregation

(MLP)

for

feature

expressiveness,

structural

it

better differentiates

as


classifier is available for multi-task ablation or interpretability overlays, although not required for the binary segmentation runs. Optimization uses AdamW (learning rate 3 x 10~*, weight decay with OneCy- cleLR for 35 epochs and early stopping on the validation Dice with a batch size of 32 and a seed of 42. During inference, probabilities are made by taking a sigmoid on the final head. Test-time augmentation

averages predictions over

flips. These design choices

are meant to stabilize learning at multiple scales, to enhance context under noise or variance, to allow for coherent, layer-aware propagation across the field, and to help sharpen crisp, layer-following boundaries at the same time mitigating speckles.

*Fig. 4. CNV detection on a representative OCT B-scan. Left: input image with subfoveal RPE elevation and subretinal hyper-reflective material (SHRM). Right: model localization heatmap (red = higher relevance) concentrated over the lesion-consistent region.*

## 5. Evaluation metrics and protocol

## 5.1. Training protocol

‘The model was trained using the AdamW optimizer with a starting learning rate of 2 x and weight decay of 1 x 107%. A batch size of 16 was used, and training was done for 20 epochs. Mixed-precision training (16-bit floating point) was used to enhance computational efficiency. An Exponential Moving Average (EMA) with decay factor 0.999 was applied to model parameters to stabilize training. To avoid

overfitting, early stopping

monitoring the validation F1-score. Model checkpointing was used to save the best-performing model during training.

In addition to the primary training configuration, several archi- tectural and optimization hyperparameters were explicitly defined to ensure stable and reproducible learning. The input OCT images were resized to 256 x 256 px and normalized using a mean and standard deviation of 0.5. Data augmentation included horizontal flipping (p = 0.5), rotations within +10°, and brightness and contrast variations of +20% to improve generalization.

For graph

grid, resulting in 256 patch nodes. The Graph Neural Network em- ployed two stacked GATv2 layers with four attention heads and a hidden dimension of 256, enabling effective relational reasoning across spatial regions.

incorporated additional param-

The weak localization framework

eters, including an erase mechanism that removes the top 15% of

ing. attention-weighted

Pseudo-mask

for

‘The

including = 0.02, Ary = 0.02, = 0.7, and = 0.2, which were determined through validation-based tuning. Further-

learning

more, the

and a margin of 0.3 to ensure effective separation of relevant and

patch

levant

## 4.9. Comparative study of CNN backbones

To further evaluate the robustness of the proposed hybrid frame- work, we conducted an additional comparative study using differ- ent CNN backbones, namely ResNet-34, DenseNet [10], and Efficient- Net [11], while keeping the GNN module fixed as GATv2. Each back- bone was integrated into the same pipeline, ensuring identical pre- processing, graph construction, training protocol, and loss functions for fair comparison. DenseNet leverages dense connectivity between layers, allowing improved feature reuse and gradient flow, which can be beneficial for capturing fine-grained retinal textures. EfficientNet, on the other hand, employs compound scaling of depth, width, and resolution, enabling a more balanced and computationally efficient architecture. The results, summarized in Table 4, indicate that ResNet-

computational ef-

34 achieves the best balance between

ficiency, and stability within the CNN-GNN framework. ResNet-34 provided more consistent feature hierarchies that translated effectively into graph representations for downstream reasoning. These findings suggest that while advanced CNN architectures offer benefits, the com- patibility between CNN feature structure and graph-based reasoning plays a critical role in overall performance.

with a patience of four epochs was used,

accuracy,

ion, each image was

into a 16 x 16

patches

during

training

shortcut

learn-

prevent

to

## 4.10. Supervised comparison study

with our

To

compare

unsuper

lesion-segmentation model is developed

UNet++ decoder with a lightweight graph module for reasoning about structure. All OCT B-scans are pre-processed to become a single-channel grayscale image, are resized to 256 x 256 px, and image intensity val- ues are normalized. Any non-zero mask pixels in the segmentation mask

are taken to be foreground, making the task binary and i

pathology type. Data is randomly split with a fixed seed into a training dataset (70%), validation dataset (10%), and test dataset (20%), with the data loader including minimal augmentation. Training uses appear- ance jitter fitting of OCT (blur and noise), photometric adjustments (gamma/CLAHE), mild elastic and affine warps, and horizontal flips.

The network backbone follows a UNet++ style encoder-decoder architecture with 4 downsampling stages and a nested decoder with deep supervision. Losses are aggregated from all heads of the net- work using weights such that later heads are emphasized, and earlier heads stabilize the gradients for training. To help localize objects in low-contrast and speckled textures, a CBAM attention module in the decoder and a context branch similar to PSP in the bottleneck are implemented. For non-local structure, features from the bottleneck are projected to a compact and latent projection, down-sampled to a regular GxG grid and treated as nodes with 4/8-neighborhood edges.

A small series of graph convolutions updates node states, and then reshape and fuse them back in with the features of the decoders by con- catenation and attention prior to the final head. An auxiliary bottleneck

(high-confidence)

thresholds

and

0.85

0.4

set

at

were ion

guidance.

tot peline, integrates a an an Attention-

was governed

by

multiple

weighting

coefficients,

loss function

used a

of 0.07

of

;

## 5.2. Explainability

Model interpretability was achieved using Gradient-weighted Class Activation Mapping (Grad-CAM). The class-discriminative localization

is calculated as:

map

where A* represents the kth feature map and af is the gradient weight:

with y* being the score for class c. The resulting heatmaps were nor- malized, thresholded (Otsu’s method), and refined via morphological operations. GNN attention weights were also analyzed to highlight structurally important patches.


*Table 1 Performance measures with ResNet + GATv2.*

|   | Metric | Value |
| --- | --- | --- |
| Precision | Test accuracy | 97.29% |
| Recall |   | 96.35% |
|   | Fl-score | 95.89% |
|   |   | 96.11% |

*Table 2 Performance across different random seeds (ResNet + GATv2).*

| Metric | Seed 7 Seed 42 Seed 123 Seed 999 Seed 2024 Mean = Std |   |   |   |   |   |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.9717 | 09719 | 09695 | 0.9742 | 09731 | 0.9721 = 0.0016 |
| Precision | 0.9594 | 09559 | 09604 | 0.9614 | 0959 | 0.9592 x 0.0019 |
| Recall | 0.9601 | 09661 | 09544 | 0.9660 | 0.9634 | 0.9620 = 0.0044 |
| Flscore | 0.959 09608 |   | 09571 | 0.9637 | 0.9612 | + 0.0022 |

*Fig. 5. OCT B-scan (left) showing RPE elevation with subretinal fluid; model localization heatmap (right, red = higher relevance) concentrated over the suspected CNV complex and adjacent subretinal space.*

## 6. Results

## 6.1. Classification performance analysis

Our primary ResNet+GATv2 model performed classification well, with an overall test accuracy of 97.29% across various categories. The performance metrics outlined in show that our model is highly discriminative across all disease categories.

These results represent a significant achievement in automated clas- sification of retinal diseases as they show that our hybrid framework combines the local feature extraction capabilities of CNNs with the global structural reasoning of graph neural networks. The balanced performance in both precision and recall metrics shows that the model effectively handles all four categories of disease without bias to a

particular class (see

standard deviation observed across multi-

The consistently low

ple random seeds demonstrates that the proposed hybrid CNN-GNN framework is stable, reproducible, and robust under different random initialization conditions (see

The visualization results show the overall interpretability methods

of our hybrid CNN-GNN framework in two representative cases of CNV. Case 1 (Fig. 4) shows an example of a typical CNV lesion with subretinal hyperreflective material forming a mound-like elevation below the RPE. The localization head appropriately localizes the pathological region with intense activation (red) specifically over the CNV region, indicating that the network has learned to focus its attention on clin- ically relevant subretinal neovascular membranes while appropriately ignoring normal retinal tissue.

5) represents a more complex pathological presentation

Case 2 (Fig.

with CNV and subretinal fluid accumulation. The attention ing scenario, mechanism as

shows

success in

activation is evident in the solid neovascular component and associated hyporeflective fluid spaces. This indicates that the model has learned to

this

*Fig. 6. Visualization results for a representative DME case. Left: input OCT B-scan showing retinal thickening and cystoid fluid spaces characteristic of DME. Right: localization head heatmap with activation concentrated over the DME region.*

*Fig.*

*a representative*

*Left: input*

*Visualization*

*results for*

*7.*

*Drusen*

*OCT*

*case.*

*showing*

*Drusen deposits.*

*Right:*

*RPE elevation*

*and*

*localization*

*head*

*B-scan*

*heatmap with activation concentrated over the Drusen region.*

*Fig. 8. Visualization results for a representative Normal case. Left: input OCT B-scan showing a healthy retina with a flat RPE layer and no pathological findings. The model produces low diffuse activation, confirming the absence*

*Right: region.*

*localization head heatmap with activation concentrated over*

*of the*

*disease.*

*Normal*

identify not only the pathological features of interest but also secondary disease manifestations that are relevant clinically for the complete CNV diagnosis.

‘The visualization pj line further supports a number of technical

achievements:

- Anatomical Accuracy - attention maps accurately focus attention on medically relevant structures and not spurious artifacts.

- Multi-Scale Integration tion and GNN spatial reasoning from entire lesions. — integrating CNN local feature extrac-

- E Clinical logical features Interpretability while explaining the heat model maps highlight decisions. specific patho-

- iv. Segmentation Quality — boundaries are defined accurately, spa- tially coherent, and capturing the entire lesion.


*Table 3 GNN variant comparison (shared ResNet-34 backbone).*

| Method | Acc. | Prec. | Recall | Fl |
| --- | --- | --- | --- | --- |
| ResNet + GATv2 | 97.20% | 96.35% | 95.89% | 96.11% |
| ResNet + GraphSAGE | 96.50% | 95.89% | 93.99% | 94.87% |
| ResNet + GIN | 95.46% | 92.94% | 95.15% | 93.79% |

*Table 4 CNN backbone comparison (fixed GATv2).*

| Backbone | Ace. | Prec. | Recall | Fl |
| --- | --- | --- | --- | --- |
| ResNet-34 + GATv2 | 97.29% | 96.35% | 95.80% | 96.11% |
| EfficientNet-BO | 95.82% + GATv2 | 94.12% | 93.45% |   |
| DenseNet-121 + GATv2 | 94.15% | 91.88% | 92.30% | 9209% |

*Fig.*

*9.*

*Effici*

*BO + GATv2*

*The left image shows the*

*original OCT B-scan, while the right image shows the corresponding attention heatmap. The activation is relatively diffuse, with broader highlighted regions extending beyond the lesion boundaries, indicating reduced spatial precision.*

- v. Robustness — the model performs consistently, with varying presentations of CNV, from simple hyperreflective lesions to ‘more complex multi strate the hybrid approach provides accurate disease classifica- tion, accurate segmentation of lesions, and clinically meaningful interpretability necessary in retinal OCT analysis. These

## 6.2. Ablation study and architecture comparison

To substantiate our decisions regarding the architecture, a series of thorough ablation studies assessing the various graph neural network variants used in our hybrid framework were performed. The compar- ison demonstrated in Table 3 that our GATv2 based methods were superior in performance:

The measured performance (GATv2 > GraphSAGE > GIN) demon- strates that attention-based graph architectures with relative position bias better spatial reasoning ability for retinal OCT analysis. possess

GraphSAGE

The

GATv2

accuracy

and 1.83% improvement over GIN, validating our architectural hypoth-

relationship

esis that

structures

by multi-head attention networks with positional encoding.

To further isolate the effect of the CNN backbone within the pro- posed hybrid framework, we conducted an additional ablation study by fixing the GNN module as GATv2 and varying the CNN encoder architecture.

The results in Table 4 demonstrate the influence of CNN backbone selection on the performance of the hybrid CNN-GNN framework. Among the evaluated architectures, ResNet-34 achieved the highest performance across all metrics, indicating that its hierarchical feature

are for graph-based reasoning. well-suited

EfficientNet-B0 [11] produced competitive results but exhibited slightly

reduced

that its

while efficient, may be less structured for graph-based relational mod-

eling. DenseNet-121 [10] showed

which can be attributed to feature redundancy introduced by dense

leading to less discriminative node embed-

dings when constructing the graph. These findings highlight that, be- yond individual model strength, the compatibility between CNN feature

representations ing

passing

and GNN-based message

in hybrid

overall

The qualitative behavior of the EfficientNet-BO [11] backbone is illustrated in Fig. 9. The model produces broader activation regions that extend beyond the primary lesion area, indicating less focused atten- tion. While the model is still able to identify disease-relevant regions, the lack of sharp localization suggests that the extracted features are less spatially structured when propagated through the graph module. ‘This behavior aligns with the slightly reduced quantitative performance observed in Table 4.

Fig. 10 shows the localization behavior of the DenseNet-121 [10] backbone. Unlike EfficientNet, DenseNet produces sharper activations;

*Fig. 10. DenseNet-121 + GATv2 localization. The left image shows the original OCT B-scan, while the right image shows the attention heatmap. The activation sharper but fragmented, with disconnected regions of high appears response.*

patchy

into multiple

however, these regions. This results in

model achieved

better

than

0.79%

attention

that

do not

form

patterns

a

coherent tributed to feature

of the lesion

Such

behavior

be at- lead-

ion

structure.

can

is captured

anatomical

for retinal

better

by dense

ing to less discriminative node embeddings and reduced effectiveness in graph-based reasoning.

In our comparison study, the ResNet + GATv2 model achieved the smoothest and most anatomically accurate segmentation, with the head producing gradients contained

a contiguous red overlay

to the CNV lesion and the decoder

that closely conformed to the lesion boundaries while not bleeding into surrounding tissue. In contrast, the ResNet + GIN variant (Fig. 11) resulted in more patchy attention maps with isolated “islands” of red activation and the decoder result, while closely capturing the main area of the lesion, showed slight fragmentation to the edge that required additional post-processing to restore connectivity.

aled features,

\+ GraphSAGE

approach

12),

In case of the ary, ResNet

(Fig.

its behavior

was

lower

heatmaps

produced

long

elongated

attention

red

bands tracing linear lesion features and the decoder mask accurately followed these red bands, although occasionally displaying slender false-positive bands extending across the retinal multilayers. In conclu- sion, while GIN and GraphSAGE provided fairly usable yet less nuanced

plays a crucial role in

spatial attention yielded the best dissected,

methods, clinically

GATv2’s

smooth

archi

valid

lesion

delineation.

## 6.3. Supervised comparison

The qualitative results in Fig. 13 accordingly demonstrate the su- pervised baselines behavior across multiple modalities of visualiza-

that the supervised baseline shown was trained and evaluated

on the OCTSk dataset, whereas the proposed model’s primary evalua- tion is conducted on the Mendeley (Kermany) dataset. The original OCT B-scan (top-left corner) shows the retinal morphology while the lesions region is highlighted. The model output (top-right corner) highlights


*Fig. 11. ResNet + GIN segmentation. Class-specific attention maps (second column) reveal patchy red activation over the lesion. The decoder mask (third column) captures most of the CNV region but shows slight fragmentation at boundaries.*

*Fig. 12. ResNet + GraphSAGE segmentation. Attention heatmaps (second column) highlight linear lesion patterns with red bands. The decoder output (third column) follows these bands to produce an elongated red overlay.*


*Table 5*

Detected: CV

*on OCTSk dataset.*

| Method | Dice (%) | ToU (%) |
| --- | --- | --- |
| UNet++ (Supervised) | 824 | 701 |
| Proposed method | 887 | 796 |

While the supervised baseline yields accurate segmentation and attention can be understood, its evaluation is limited by the need maps for pixel-level annotations and the assumption or reliance on a fixed training distribution during training. However, unlike the above, our unsupervised approach removes the dependency on labeled data since the model is able to learn adaptively from the OCTSk dataset [48] without additional annotation cost. But the more critical advan- any tage of the unsupervised approach is its improved ability to segment lesions in cases where the supervised model fails due to either over- segmentation or improper localization due to the presence of fluid and unclear boundary conditions. The unsupervised approach is capable of global structure reasoning along with consistency in the context which helps the model to retain the global structure context while preserving the local precision in the resulting segmentation masks.

that the possibility of comparing

However, it should be

mentioned model to

the proposed

‘models

ion

sup

with large capacities, such as TransUNet and Swin-UNet, is limited by the lack of annotations in the utilized data sets. The major Kermany OCT data set, which consists of over 84,000 images, lacks pixel-level annotations. The OCTSk dataset, while annotated, contains approxi- mately 5000 images - a scale insufficient for stable convergence of transformer-based segmentation models that typically require tens of thousands of annotated slices to generalize reliably. Under these con- straints, UNet++ is employed as a representative strong supervised segmentation baseline. Critically, the proposed method achieves a Dice score of 88.7% and IoU of 79.6% on OCTSk without utilizing any pixel-level supervision during training, outperforming the fully super- This vised UNet++ baseline by 6.3 and 9.5 percentage points, respectively. that struct weakly learning can achieve competitive or superior localization quality compared to fully supervised segmentation approaches.

Combined

Attention

*GN structural Attention*

*Fig. 13. Supervised baseline outputs on OCT scan showing CNV localization, probability map, binary mask, and attention visualizations, trained and eval- uated on the OCTSk dataset.*

Quantitative evaluation on the OCTSK test set further supports these observations. As shown in Table 5, the supervised UNet++ baseline achieves a Dice score of 82.4% and IoU of 70.1%, while the pro- posed method attains 88.7% Dice and 79.6% loU, indicating improved segmentation consistency and overlap with ground truth annotations.

the area for CNV, denoted as a region in red, which is clearly delineated with a boundary that closely follows the contours of the retinal layers. ‘This visualization confirms that the network is capable of localizing and

## 6.4. Loss functions

A composite objective function was designed to balance classifica-

tion with effective weak localization.

accuracy

delimiting the pathological subretinal domain accurately to anatomy.

(middle-left) estimates the

The

map

con-

fidence of the network, with the majority of values for the lesion region very close to unity. The corresponding binary mask (middle- right) is produced from Otsu thresholding producing an estimated lesion area of 72.8%. This is a large proportion of the entire scan, but also consistent with the strong posterior distribution observed in the probability heatmap, suggesting that the network is confident in detection of the impacted region. The attention visualizations provide additional interpretability. The GNN structural attention (bottom-left corner) emphasizes layer-level continuity with activation concentrated

## 6.4.1. Classification loss (cross-entropy)

For a batch of N samples, the cross-entropy loss is defined as:

where y,, is the ground truth label for class c, and ,, is the predicted probability.

## 6.4.2. Attention entropy loss

diverse and

To

ized entropy of attention weights [46] were minimized:

where a, is the attention score for patch j in sample i, M is the total number of patches, and ¢ is a small constant.

suggests that the GNN module

along

layer

retinal

boundaries.

This

fal attention the normal-

provides

enforcing

towards

maps,

structure

some

maintaining consistent layer-wise segmentation. Alternatively, the joint CNN+GNN attention (bottom right) uses fine-grained local cues from the CNN with a global relational context provided by the GNN for saliency distribution that highlights both the lesion body and the subtle transitions with the boundary. The two modalities are illustrative of how structural reasoning can be used alongside pixel-level encoding to form smooth and contextually relevant segmentation.


*Table 6*

## 6.4.3. Total Variation (TV) loss

To enforce spatial

edge-weighted TV loss [45] was used:

*Loss analysis of the model.*

in the weak

maps, an

| Loss type | Value |
| --- | --- |
| Total loss | 0.197 |
| Classification CE | 0.107 |
| Localization CE | 0.088 |
| MIL loss | 0.089 |
| TV loss | 0.0001 |
| Spatial entropy | 0.017 |
| Attention entropy | 0.027 |
| Segmentation consistency |   |

where , denotes the mask intensity at pixel (u,v), and w?, and wf,

m,

are image-gradient based weights that down weight edges aligned with strong image

## 6.6. Performance assessment

metrics to reflect different aspects was of model assessed performance. through Accuracy multi-

ple

was used as a measure of overall correct classification, which is the Proportion of images classified into the correct diagnosis. However,

git po cy g iven the potential for accuracy to be misleading with class imbalance, macro-averaged metrics were reported in addition to accuracy to assess performance with reasonable fairness across all disease classes. The macro-averaged precision summarized the ability of the model to avoid false positives while also taking into consideration how reliably positive

predictions corresponded to

the sensitivity by measuring how often it was able

to identify the diseased case correctly across the disease classes. Finally, the macro-averaged Fl-score, which is the harmonic mean between precision and recall, provided a single balanced expression of the models overall effectiveness to identify as well as differentiate between

Collectively,

the diseases.

solid overall

of classificati and fairness

ion

accuracy

the proposed

Table

7

compares

several recent deep learning approaches evaluated on the Kermany Mendeley OCT dataset. Lightweight CNN-based methods such as Bhan- dari et al. [58] and transfer learning approaches including Jaimes et al. [5] and Barbosa et al. [4] achieve strong classification per- formance, while transformer-based architectures such as the OCCT Transformer [59] and Vision Transformer variants proposed by Aka

improve retinal disease recognition through attention-

et al. [3] further based modeling.

The proposed ResNet + GATv2 framework achieves the highest

accuracy of 97.31% while maintaining competi-

overall classification

tive precision, recall, and F1-score across all disease categories. These findings indicate that integrating graph-based relational reasoning with CNN feature extraction enables effective modeling of structural depen- dencies within retinal OCT images. In addition to strong quantitative performance, the proposed framework provides enhanced anatomical interpretability through graph attention and weakly supervised local-

isualization of di

ization, enabling clinically retinal regions.

All compared methods reported in Table 7 were evaluated on the same Kermany/Mendeley OCT dataset under comparable evaluation settings and standard performance metrics to ensure fair and consistent comparison.

## 6.4.4. Dice loss

For weak

partial cross-entropy [57] was employed:

where and are the predicted and ground truth pseudo mask values,

p, g;

respectively.

Dice loss in combination with

p

,

## 6.4.5. Contrastive sufficiency and necessity losses

To ensure patch-level

essential features, contrastive losses were introduced.

Sufficiency: ensures top-k attended patches alone can predict the correct class. Necessity: ensures removing top-k patches degrades per- formance.

where sim(z, z*) is cosine similarity, is a temperature hyperparam- eter, z* is a positive (sufficient) representation, and =~ are negatives (non-relevant patches) [50].

The final objective was expressed as:

with A coefficients empirically tuned.

The 4 coefficients were determined through manual tuning based on validation performance. Multiple candidate values were empirically evaluated across training runs, and the final configuration (e.g., iry = 0.02) was selected to achieve the best trade-off between classification

quality. The model exhibited moderate sensi-

and localization

accuracy

tivity to the choice of 4 values, particularly for Az and while

remaining stable under the selected parameter settings.

the true class. The macro-averaged recall

captured discriminative and

these complementary

ensured

measures

a

with

for

considerations

all diagnosis

classes.

across

framework with

ResNet

\+ GATv2

meani

## 6.5. Multi-component loss function analysis

The loss function includes multiple objectives in order to opti-

mize

across the models. The overall model loss combines classification cross- entropy (weight = 1.0), localization cross-entropy (weight = 0.5), MIL attention loss (weight = 0.5), total variation regularization (weight = 0.1), spatial entropy regularization (weight = 0.05), segmentation consistency loss (weight = 0.2), and attention entropy loss (weight = 0.05). Our model achieved low loss values across every loss component when being tested as shown in Table 6.

The lower loss values produced when testing the models across

every component indicates that our multi-objective optimizati

loss

was successful in balancing competing objectives without degrading performance in any single task performance.

across i ion and

ion objectives

## 6.7. Interpretability and visualization results

One of the critical advantages of our method is its interpretabil- ity mechanisms that provide clinically meaningful interpretations of modeling decisions. Included visualization pipeline illustrates multiple

ive methods that are

i

- ii. anism effectively identifies disease-affiliated regions with high accuracy. In cases of CNV, the model consistently attends to Localization Head Attention Maps: The spatial attention mech-


*Table 7 Comparison with external State-of-the-Art methods on Kermany Mendeley OCT Dataset.*

| Method | Year | Ace (%) | Prec (%) | Recall (%) | FL |
| --- | --- | --- | --- | --- | --- |
| Bhandari et al. (Lightweight CNN) [55] | 2023 | 94.29 | 9450 | 9429 | 94.30 |
| Jaimes et al. (VGG-16 Transfer Learning) [5] | 2025 | 95.19 | 95.29 | 95.19 | 95.20 |
| Barbosa et al. (VGG-19 Fine-tuned) [4] | 2025 | 95.20 | 95.43 | 95.20 | 95.21 |
| Chowa et al. (OCCT Transformer) [59] | 2024 | 97.00 | 96.52 | 97.27 | 96.57 |
| Akea et al. (VIT Comparative) [3] | 2024 | 96.07 | 95.80 | 96.07 | 95.93 |
| Proposed ResNet + GATv2 (Ours) |   | 7.31 | 95.90 | 96.34 | 96.12 |
|   | - |   |   |   |   |

subretinal hyperreflective material and neovascular complexes. ‘This is evidenced by the attention maps, which show significant attention is placed on pathological areas, while also demonstrat- ing a competent appreciation of the structures within the retinal layers.

- i. Class-Specific MIL Attention: The class-specific attention mecha- nism yields a unique attention profile for each disease definition, thereby leveraging fine-grained localization of pathological fea- tures. The attention visualizations provide an interesting finding that the model has learned to associate specific spatial patterns in to feature phenotypes. space

- iii. Decoder Segmentation Maps: The U-Net decoder results in high- quality segmentation masks that distinguish lesions with ap- propriate precision. The spatial response of the segmentation output shows a strong spatial coherence and an anatomical plausibility in how contours are assigned to lesions, including smooth and

- = Post-processed Mask Generation: The clean binary masks op- timizing for clinical quantification are a result of our post- processing pipeline, including Gaussian blur, Otsu thresholding, ‘morphological operations, and connected component analysis. The masks have high fidelity to pathology while minimizing artifacts that would confuse the disease.

## 6.8. Model architecture efficiency and scalability

The hybrid model demonstrates computational efficiency with 22.2 million trainable parameters. The chosen configuration represents an appropriate trade-off between model capacity and computational costs. The breakdown, according to the architecture of the deep learning model, would be the ResNet-34 encoder (21.3M parameters), GATv2 graph network (0.7M parameters), MIL attention modules (0.15M pa-

rameters), and decoder

(0.05M

of training happened after 20 epochs, peak validation accuracy being recorded on epoch 18. The experimental mixed precision training (16-bit) reduced memory utilization of the training process by approx- imately 50%, while providing a memory safe operation. The average per-image inference time was 45 ms when utilizing GPU hardware, providing a viable clinical real-time application.

[¢

## 6.9. Robustness and generalization analysis

Experimental evidence suggests strong generalizability traits on test dataset results. Our model provided consistent performance across variation in the quality of the OCT image, contrast, and noise intro- duced by the testing dataset. The precision-recall trade-off was well balanced(precision: 96.35%, recall: 95.89%) suggesting that both false positive and false negative biases were avoided, an important aspect

for clinical settings where false negatives visualizations and false positives

Attention

ically sound feature representations were learned by the model. The model used medically-relevant structures rather than spurious image artifacts. This suggests that the hybrid CNN-GNN architecture success- fully captures the underlying pathophysiology of retinal diseases and does not simply memorize dataset-specific patterns.

represent

that

suggest

anatom-

## 7. Future work and limitations

Even though our hybrid framework combining CNN and GNN has demonstrated strong classification accuracy and lesion segmentation, there are several more limitations that need to be examined in future work before a “clinically ready’ label can be placed on the approach. First, in our experiments, a single public OCT dataset was used, which may not have covered the full range of imaging protocols, device- specific artifacts, or patient demographics that clinicians encounter in daily practice. The concern here is the risk of bias in the dataset and ultimately model overfitting. Second, constructing the graph on an anchor point-to-point static grid may not capture variable anatomical relationships present in complex pathology cases, inhibiting spatial rea- soning in potentially subtler disease states. Third, both inference speed and resource consumption have yet to be optimized for deployment on standard PACS workstations or commonly available portable OCT devices. Finally, while our attention maps and lesion segmentation masks us to capture some interpretability, this study currently allow do not provide a quantitative measure for uncertainty that could offer

to their predictions. Overcoming these

clinicians added confidence

issues by implementing the strategic growth of multi-center databases, model dynamic and hierarchical anatomy and mapping, including implementing quantifica- real-time tion would be some of the key steps needed to make sure that this technology is implemented as a reliable diagnosis tool for clinicians.

Future directions for this research project are set to build upon the current success by broadening and deepening the scope of our hybrid CNN-GNN architecture, moving beyond proof of concept toward full clinical application. In order to rigorously validate the robustness and generalizability of the model in real-world clinical contexts, it

necessary to have a larger training set that takes into account

will be

multicenter data sets, which take into consideration different patient

and OCT i vendors. The

ions, imaging

iques,

longitudinal OCT image data combined with patient information not only will enable the temporal analysis of the disease state and progres- sion but will also allow for predictive modeling of patient outcomes for personalized treatment plans. To continue these advancements, it will be useful to incorporate other imaging modalities, such as fundus imaging and OCT angiography, along with patient metadata, which will

serve as the basis for a multimodal diagnostic aid system.

Several multi-modal learning

images, and metadata of the patients have been found successful in im- proving the accuracy rate in diagnosing glaucoma. Further studies may consider applying multi-modal learning in the suggested framework for ture use.

From an architectural perspective, there is much scope for im-

beyond the grid-based

network. Instead, learning graph-based relationships that have meaning based on the visual relationships within the OCT images will make it

complex

easier

linking

architectures

even more depth. When it comes to implementation, reducing infer- ence time and resource consumption will be imperative to smoothly integrate into established picture archiving and communication systems (PACS). Finally, building uncertainty quantification parameters that convey predictive confidence will justify clinician consideration of Al

OCT, fundus

of the graph neural

graph

associations,

to detect

anatomical

and

hierarchal

tissue to global

local

retina

would

add

structure


recommendations, and hopefully lead to a shared diagnostic platform that enhances, rather than replaces, the knowledge of ophthalmologists.

Although the proposed framework demonstrates strong ison

classifica-

direct

fully

tion and weak

with

supervised segmentation architectures remains limited by the absence of pixel-level annotations in the primary Kermany OCT dataset. Future work will focus on evaluating the proposed framework on larger fully annotated retinal OCT segmentation benchmarks and extending the ap- proach towards multi-modal retinal diagnosis integrating OCT, fundus imaging, and clinical metadata.

## 8. Conclusion

This study presented a hybrid framework that integrates local texture modeling (CNN/ResNet-34) with global structural reasoning (GATv2 on a patch graph) and weak localization heads for OCT analysis. For the CNV, DME, Drusen, and Normal classes, the model achieved 97.29% test accuracy with a balanced precision/recall, and produced interpretable medical heatmaps and weak segmentation. The ablation study suggests that attention-based graph message passing with relative positional bias (GATv2) is superior for spatial reasoning compared to GraphSAGE and GIN, resulting in smoother, anatomically consistent attention and clearer delineation of lesions. Our processing pipeline, which included mixed-precision training, EMA stabilization, early stopping, and TensorBoard logging, provided both efficiency and reproducible procedures with average inference latency of 45 ms per B-scan showing promising potential for clinical workflows. A cen- tral contribution is its label-efficiency: the weak/unsupervised design

can generate class-specific

annotation), thus addressing both cost and consistency issues in ges erating fully dense(pixel-wise) labels while providing interpretabil- ity. By combining CNN features with graph-based context and using erased-MIL to discourage shortcut learning, this study also shows that the model remains robust to variability in presentation (e.g., fluid and boundary ambiguity) while avoiding over-reliance on small dis- criminative regions. The results show, along with supervised baseline comparison, that structure-aware, label-efficient learning can achieve similar, if not superior, behavior to a supervised state for lesion lo- calization whilst maintaining high diagnostic accuracy, together sup- porting connectivity-aware models as a viable path towards scalable, contextually trusted OCT analysis in a clinically setting.

without pixel

## CRediT authorship contribution statement

Aryaman Chandra: Writing — review & editing, Writing original draft, Visualization, Validation, Software, Resources, Project admin- istration, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Aniket Choudhury: Writing — review & editing,

Writing original admi draft, Visualization, Validation, igation, Software,

Project

Data curation, Conceptualization. Prateek Sharma: Writing — review & editing, Writing original admi draft, Visualization, Validation, Software,

Resources, Project

analysis, Data curation, Conceptualization. Kavya Singh: Supervision. Reenu Rani: Supervision.

Resources, analysis,

Formal

Formal

## Funding

“This research did not receive any specific grant from funding agen- ies in the public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The authors declare that they have no known competing finan- cial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgment

‘The authors

express

, Vellore,

successful completion of this work. We also gratefully acknowledge

the

of Science and

providing access to the High-Performance Computing (HPC) cluster

under grant number SR/FST/MS-11/2023/139(C)-VIT Vellore, which was essential for the computer simulations. Additionally, the authors thank Dayasagar Parida for his valuable assistance with the numerical simulations.

gratitude

their

sincere

Vellore

Institute

of

to

providing

for

the

towards

the

support

necessary

of India, for

## Data availability

The kaggle. ‘datacets, OCT [47] classification t-images-s dataset publicly available Mendeley 1 1

studies.

pli ed/data) is used in the oCTSk [48] (h Kly sup ‘doi.org/10.1763

uses

2), which includes

graded retinal layer annotations. Both datasets are publicly available

purposes. In this study, no additional human-subject data

for research

were collected and no

citation (including DOI) of the dataset are provided in the details Reference section.

and includes multi

scans

private clinical data were used. The complete

## References

- [1] H. Gu, Y. Zhou, J. Zhao, H. Zhang, X. Pan, B. Li, B. Zhang, Y. Wang, S. Xia, H. Lin, J. Wang, D. Ding, X. Li, S. Wu, J. Yang, and generalizability of deep learning for multiple retinal diseases using bimodal imaging of fundus photography and optical coherence tomography, Front. Cell Dev. Biol. 13 (2025) 1665173, 2025.1665173. formance

- [2] U. current and future role of optical coherence tomography in the management of age related macular degeneration, Eye 31 (2017) 26-44, hitp://dx.doi.org/10. 1038/eye.2016.227. S. Klimscha, SM. Waldstein, H Bogunovié, A view of the

- [3] S. Akea, Z Garip, E. Ekinci, F Atban, Automated classification of choroidal neovascularization, diabetic macular edema, and drusen from retinal OCT using vision transformers: a comparative study, Lasers Med. Sci. 39 (1) (2024) 140, images

- [4] G. Barbosa, E Carvalho, A. Guerra, S. Torres-Costa, N. Ramido, M.LP. Par- ente, M Deep leaning to distinguish edema secondary to retinal vein occlusion and diabetic macular edema: a multimodal approach using OCT and infrared imaging, J. Clin. Med. 14 (3) (2025) 1008, hitp://dx.doi.org/10.3390/ jem14031008.

- [5] from OCT images using a VGG16 and transfer Detection iscov. of Appl. retinal Sci. diseases 7 (3) Jaimes, W.J. Arenas, HJ. Navarro, M Altuve, (2025) 1-14, hitp://dx.doi.org/10.1007/542452-

- [6] S. retinal damage detection, in: 2023 10th EEE Uttar Pradesh Section image Conference Pp. 869-875, Sathyan, analysis: M. Chanchal, on hitp://dx.doi.org/10.1109/UPCON59197.2023.10434663. Electrical, Electronics and Computer Engineering, UPCON, 2023, Exploring S. Raghul, convolutional G Jeyakumar, neural Advancing network model optical tomography ve

- [7] SPK. Karri, D. Chakraborty, J Chatterjee, Transfer learning based classification age of optical related coherence macular degeneration, tomography images Biomed. with Opt. diabetic Express macular 8 (2) (2017) edema 579-592, and dry hitp://dx.doi.org/10.1364/BOE.8.000579.

- [8] AG. Roy, S. Conjeti, SPX. Karri, D. Sheet, A. Katouzian, C. Wachinger, coherence N. ReLayNet Nava, Retinal layer and fluid segmentation of macular optical phy using fully convolutional networks, Biomed. Opt. Express 8 (8) (2017) 3627-3642, hitp://dx.doi.org/10.1364/BOE.8.003627.

- [9] PK. Kam, WH. Abdulla, Precision segmentation of subretinal fluids in OCT using hutp://dx doi. attention-based U-net, Bioengineering 1101032. 11 (10) (2023) 1032, multiscale

- [10] G. Huang, Z Liu, L van der Maaten, K.Q. Weinberger, Densely connected convolutional networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2017, pp. 4700-4708, hitp://dx.doi.org/ 10.48550/arXiv.1608.06993.

- neural networks, in: Le, Rethinking of the 36th model International scaling for Conference convolutional on M. Tan, QV. EfficientNet Machine ICML, 2019, hutp://dx.doi.org/10.48550/arXiv.1905.11946, PMLR 97:6105-6114.


- nz) LD. Mienye, § Graph neural networks in medical imaging: Methods, applications and future doi.org/10.3390/info16121051. Information 16 (12) (2025) 1051, hutp://dx.

- ns) G. Cao, Y. Wu, Z. Peng, Z Zhou, C Dai, Selfattention CNN for retinal layer segmentation in OCT, Biomed. Opt. Express 15 (3) (2024) 1605-1617, hp: //dx.doi.org/10.1364/BOE.510464.

- M. Brinkmann, M. Concilio, M.D. Toro, R Rejdak, Residual self attention vision transformer for detecting acquired vitelliform lesions and age-related macular drusen, Sci. Rep. 15 (2025) 17107, 02299-y. ‘Skublewska-Paszkowska, K. Nowomiejska, B. o,

- Mm. Oghbaie, T. Arajo, U. Schmidt-Erfurth, H. Bogunovié, Vifatrollout: fully transformer-based classifier for retinal OCT volumes, Comput. Med. Imaging Graph. 118 (2024) 102452, 102452. compmedimag 2024.

- 6] N. CL Mehta, Tsai, J. Yang. G. ided Demirci, weakly X. supervised Hu, M.S. lesion 2024.103139. images, Image Anal. 94 (2024) 103139, hitp://dx.doi.org/10.1016/j media. Med. M. Naguib, on retinal C. Chen, OCT

- j.compbiomed.2022.106467. X. Liu, Q. Liu, Y. Zhang, M. Wang, J Tang, TSSK-net: Weakly supervised biomarker localization and segmentation with image level annotation in retinal OCT images, Comput. Biol. Med. 153 (2023) 106467, hitp://dx.doi.org/10.1016/

- ns) vision graph convolutional network for te tomography Opt. Express 16 (6) (2025) 2312-2326, hitp://dx.doi.org/10.

- no) an CAM: in: org/10.1109/ICCV.2017.74. Proc. visual IEEE explanations Int. Conf. Comput. from deep Vis., networks ICCV, 2017, via gradient-based pp. 618-626, hip://dx doi. M. Cogswell, A. Das, R. Vedantam, D. Parikh, D Batra, Grad-

- 120] s. Brody, U. Alon, E Yahav, How attentive are graph attention networks? in: International Cor on Representations, ICLR, 2022,

- 21] 00002-9. H. Xiao, G. Yang, Z. Li, C Yi, GNNs surpass transformers in tumor segmentation, Sci. Rep. 15 (2025) 1388, i 02s

- 122) M. Ravinder, G. Saluja, S. Allabun, M.S. Algahtani, M. Abbas, M. Othman, BO. Soufiene, Enhanced brain tumor classification using graph convolutional neural network, Sci. Rep. 13 (2023) 15098, 023-41407-8.

- [23] W. Wu, X. Liu, RB. Hamilton, AA. Suriawianata, S. Hassanpour, Graph con- volutional neural networks for histologic classification of pancreatic cancer, Arch. Pathol. Lab. Med. 147 (7) (2023) 851-860, hitp://dx.doi.org/10.5858/

- [24] L. Tognetto, OCT-based deep-learning models for the sans Inferrera, Sci. Rep. L. Borsatti, 13 (2023) A. 14628, Miladinovic, hitp://dx.doi.org/10.1038/541598-023-41362- D. Marangoni, i R. Giglio, of A. Accardo, retinal key D

- 125] N. A comprehensive review of artificial intelligence models for retinal diseases Werghi, Artif. Intell. Rev. 57 (2024) 111, hutp://dx.doi.org/10. T. Hassan, MU. Akram, HL. Raja, AA. Abd-Alrazaq, S. Yousefi, ‘major

- 126 J. Yang, C. Hsu, J. Wang, B. Wu, Y. Lu, Y. Ding, Z. Zhao, K. Tang, F. Lu, L Ma, Hyreti-net: hybrid retinal diseases classification and diagnosis network using optical coherence tomography, Fron. Med. 12 (2025) 1660920, hitp: //dx.doi.org/10.3389/fmed.2025.1660920.

- 1271 MA. Hussain, A. Bhuiyan, CD. Luu, RT. Smith, RH. Guymer, H. Ishikawa, J.5. Schuman, K Ramamohanarao, Classification of healthy and diseased retina using SD-OCT imaging and random forest algorithm, PLOS ONE 13 (6) (2018) 0198281, journal. pone.0198281.

- 128] CS. Lee, AJ. Tyring, NP. Deruyter, Y. Wu, A. Rokem, AY. Lee, Deep- learning based, automated segmentation of macular edema in optical coherence tomography, Biomed. Opt. Express 8 (2017) 3440-3448, 1364/BOE.8.003440. 10.

- 129] Sikei, E. Rumetshofer, N. Schmidinger, A. Mayr, U. Schmidt Erfurth, H. Bogunovié, G Klambauer, Multi-modal representation learning in retinal imaging using self supervised learning for enhanced clinical (2024) 26802, hitp://dx.doi.org/10. Sci. Rep. 14

- 130] J. learning 1342-1350, http://dx.doi.org/10.1038/541591-018-0107-6. De Fauw, for diagnosis JR. Ledsam, and B. referral Romers in Paredes retinal frie Noture Med. 24 le (2018) deep

- 31] S. Shim, M.S. Kim, C.G. Yae, Y.K. Kang, JR. Do, SK. Kim, HL. Yang, Develop- ment and validation of a multi-stage self-supervised learning model for optical coherence tomography image classification, J. Am. Med. Inf. Assoc. (JAMIA) 32 (5) (2025) 800-810, http://dx.doi.org/10.1093/jamia/ocaf021.

- 1321 TW. OCT scans, journal PLOS pig, 0000411. Digit. Health 3 (8) (2024) e0000411, hitp://dx.doi.org/10. 1371 Mehanian, crimination Wheeler, Self of full supervised K. thickness Hunter, contrastive PA. macular Garcia, holes learning H. from Li, improves AC. Thomson, machine ‘membranes A. learning Hunter, in retinal dis- C

- [33] FE. Jannat, S. Gholami, MN. Alam, H Tabkhi, OCT-SeliNet: a selfsupervised framework with multi-source datasets for generalized retinal disease detec- tion, Front. Big Data 8 (2025) 1609124, 1609124. 2025.

- [34] L Fang, D. Cunefare, C. Wang, RH. Guymer, S. Li, § Farsiu, Automatic segmentation of nine retinal layer boundaries in OCT images of non-exudative AMD patients using deep learning and graph search, Biomed. Opt. Express 8 (5) (2017) 2732-2744, hitp://dx.doi.org/10.1364/BOE.8.002732.

- [35] T. Schlegl, SM. Waldstein, H. Bogunovic, F. Endstrafer, A. Sadeghipour, AM. Philip, D. Podkowinski, BS. Gerendas, G. Langs, U Schmidt-Erfurth, Fully automated detection and quantification of macular fluid in OCT using deep learning, Ophthalmology 125 (7) (2018) 1134-1144, htp://dx.doi.org/10.1016/ 2017.10.031.

- [36] S. Mukherjee, T.D. Silva, C. Duic, AT. Thavikulwat, E. Chew, C. Cukras, G. domain segmentation OCT devices, algorithms Ophthalmol. for age elated Sci. 3 (2025) 100670, htp://dx.doi.org/10. ion with 2 spectral Jayakar, T.D. L Keenan, Validation of deep learning based automatic retinal layer

- [37] S. Sreng, py P. Ramesh, Husain, P.DN. Schmetterer, Phuong, NEBA. D Wong, Gani, Widefield J. Chua, OCT M.E. volumetric Nongpiur, T segmentation using 15 (2025) 6676, hitp://dx.doi.org/10.1038/541598-025-89476-1. CNN and integration, Sci. Rep.

- 138] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, [39] 2015, A. Nair, H. Arvidsson, J.EV. Gatica, N. Tudzarovski, K. Meinke,

- A graph neural network framework for mapping histological topology in oral mucosal tissue, BMC Bioinformatics 23 (1) (2022) 506, hutp://dx.doi.org/10. 1186/512859-022-05063-5. Sugars,

- [40] HL convolutional Li, J. Yang, network Z Xuan, for M. ultrasound Qu, Y. Wang, C Feng, A spatio-temporal ic landmark media. graph 2024. Med. mage Anal. 97 (2024) 103272, 103272.

- [41] A. Zedadra, O. Zedadra, M.Y. Salah Salah, A Guerrieri, Multi-modal AI for multi- label retinal disease prediction using OCT and fundus images: a hybrid approach, Sensors 25 (14) (2025)

- [42] //dx.doi.org/10.3390) jimaging9020046. K mentation: Alomar, a HL survey Aysel, and 4492, X new Cai, strategies, Data augmentation J. Imaging in 9 classification (2) (2023) 46, and seg-

- [43] J. Kugelman, D. Alonso-Canerio, S.A. Read, S.J. Vincent, M.J. Collins, Enhancing OCT patch-based segmentation with improved GAN data augmentation and semi supervised learning, Neural Comput. Appl. 36 (2024) 18087-18105, hitp: //dx.doi.org/10.1007/500521-024-10044-1.

- [44] S. Manikandan, R. Raman, R. Rajalakshmi, S. Tamilselvi, RJ. Surya, Deep learning based detection of diabetic macular edema using optical coherence (2023) 1785-1796, and fundus images: A meta-analysis, Indian J. Ophthalmol. 71 (5) http://dx.doi.org/10.4103/LJ0.1J0 2614 based 22. noise

- [45] LL algorithms, Rudin, S. Phys. Osher, D: Nonlinear E. Fatemi, Phenom. Nonlinear 60 (1-4) total (1992) variation 259-268, hitp://dx.doi. removal

- [46] Z Zhang, L. Lin, Y. Cao, H Zhao, Interpretable deep learning via attention entropy minimization, in: NeurIPS Workshop on Interpretable Machine Learning, 2021, pp. 1-7, http://dx.doi.org/10.48550/arXiv.2112.04580.

- [47] DS. Kermany, M. Goldbaum, W. Cai, C.C. McKeown, G. Yang, X. Wu, F. Yan, J. Dong, MK. Prasadha, J. Pei, M.Y.L. Ting, J. Zhu, C. Li, S. Hewett, J. Dong, L Ziyar, K Zhang, Identifying medical diagnoses and treatable diseases by image-based deep leaming, Cell 172 (5) (2018) 1122-1131, hitp://dx.doi.org/10.1016/.cell.2018.02.010, .e9. H. Liang, SL. Baxter, A.

- [48] Arikan, M. OCTSk, A Dataset of and Multi-Graded Annotations for Retinal Layers, University College London, 2023, hitp://dx.doi.org/10.5522/04/ 22128671.v4, Dataset.

- [49] Y. Abbas, HJ. Hadi, K. Aziz, N. Ahmed, MU. Akhtar, MA. Alshara, P Chakrabarti, Reinforcement based leveraging transfer learning for multiclass optical coherence tomography images classification, Sci. Rep. 15 (2025) 6193,

- [50] Y. Zhang, X. Yuan, L. Luo, Y. Yang, S. Zhang, C Xu, Local contrast learning for one-shot learning, Appl. Sci. 14 (12) (2024) 5217, htp://dx.doi.org/10.3390/ app14125217.

- [51] M. Ilse, JM. Tomezak, M Welling, Attention based deep multiple instance learning, in: Learning, 80:2127.2136. ICML, 2018, hitp://dx.doi.org/10.48550/arXiv.1802.04712, PMLR of the 35th International Conference on Machine

- [52] KK. Singh, Y.J. Lee, Hide-and seek: Forcing a network to be meticulous for ‘weakly-supervised object localization., in: Proceedings of the IEEE International Conference on Computer Vision, ICCV, 2017, pp. 3544-3553, hitp://dx.doi.org/ 10.1109/1CCV.2017.380.

- [53] P. Pandey, M. Chasmai, T. Sur, B. Lall, Robust prototypical few-shot organ segmentation 2490-2501, with regularized neural odes, IEEE Trans. 2023.3258069. Med. Imaging 42 (8)

- [54] (2023) Z Li, Y. coherence 203, hutp://dx.doi.org/10.3390/jimaging9100203. Han, tomography images with swin transformer V2, J. Imaging 9 (2023) X Yang, Multi-fundus diseases classification using retinal optical


- [55] H. Xu, X. Zhang, R. Lv, Adcformer: a hybrid model of adaptive depth-wise convolution and transformer for retinal macular edema segmentation in OCT images, Signal Process. Control. 107 (2025) 107949, Biomed.

- [56] H. Xu, X. Zhang, R Lv, SAFP-net: A spatially augmented feature pyramid network for automatic detection of microaneurysms, Biomed. Signal Process. Control. 100 (2025) 106998, bspe.

- [57] F. Milletari, N. Navab, S.-A Ahmadi, V-net: fully convolutional neural for volumetric medical image segmentation, 2016, arXiv.1606.04797, arXiv preprint arXiv:1606.04797.

- [58] M. Bhandari, T.B. Shahi, A Neupane, Evaluating retinal disease diagnosis with an interpretable eight CNN model resistant to adversarial attacks, J. Imaging 9 (10) (2023) 219, htp://dx.doi.org/10.3390/jimaging9100219.

- [59] SS. Chowa, MRL Bhuiyan, LJ. Payel, A. Karim, LU. Khan, S. Montaha, MZ. Hasan, M. Jonkman, S. Azam, A low complexity efficient deep learning model for automated retinal disease diagnosis, J. Heal. Informatics Res. 9 (2025) 1-30, hitp://dx.doi.org/10.1007/541666-024-00182-5.
