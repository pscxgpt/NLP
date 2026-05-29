Intelligent Medicine 2 (2022) 161–173
Contents lists available at ScienceDirect
Intelligent Medicine
journal homepage: www.elsevier.com/locate/imed
Review
A survey of automated International Classification of Diseases coding:
development, challenges, and applications
Chenwei Yan 1, 2 , Xiangling Fu 1 ,2 , ∗ , Xien Liu 3, ∗ , Yuanqiu Zhang 1 ,2 , Yue Gao 1 ,2 , Ji Wu 3, Qiang Li 4
1 School of Computer Science, Beijing University of Posts and Telecommunications, Beijing 100876, China
2 Key Laboratory of Trustworthy Distributed Computing and Service (BUPT), Ministry of Education, Beijing 100876, China
3 Department of Electronic Engineering, Tsinghua University, Beijing 100084, China
4 Beijing Tsinghua Changgung Hospital, Beijing 102218, China
a r t i c l e i n f o a b s t r a c t
Keywords: The International Classification of Diseases (ICD) is an international standard and tool for epidemiological in-
International Classification of Diseases coding vestigation, health management, and clinical diagnosis with a fundamental role in intelligent medical care. The
Disease classification assignment of ICD codes to health-related documents has become a focus of academic research, and numerous
Health-related document
studies have developed the process of ICD coding from manual to automated work. In this survey, we review the
Electronic medical record
developmental history of this task in recent decades in depth, from the rules-based stage, through the traditional
Medical record management
Clinical coding
machine learning stage, to the neural-network-based stage. Various methods have been introduced to solve this
problem by using different techniques, and we report a performance comparison of different methods on the pub-
licly available Medical Information Mart for Intensive Care dataset. Next, we summarize four major challenges of
this task: (1) the large label space, (2) the unbalanced label distribution, (3) the long text of documents, and (4)
the interpretability of coding. Various solutions that have been proposed to solve these problems are analyzed.
Further, we discuss the applications of ICD coding, from mortality statistics to payments based on disease-related
groups and hospital performance management. In addition, we discuss different ways of considering and evaluat-
ing this task, and how it has been transformed into a learnable problem. We also provide details of the commonly
used datasets. Overall, this survey aims to provide a reference and possible prospective directions for follow-up
research work.
1. Introduction rect ICD code assignment [1] . Another survey report indicated that the
cost of correcting wrong codes of ICD-9-CM could reach US $25 billion
The International Classification of Diseases (ICD) is an internation- per year [2] . Thus, coding errors not only have an impact on patients’
ally unified disease classification method developed by the World Health follow-up treatment but also cause significant losses to the medical ex-
Organization. It classifies diseases according to their etiology, pathol- pense payment system. Therefore, from a practical perspective, ICD is
ogy, clinical manifestations, and anatomical location in a systematic of great significance as a means of improving the efficiency of medical
fashion, using coding to represent diseases. Assigning the appropriate record management, reducing the cost of DRGs, and minimizing loss in
ICD codes to health-related documents is called ICD coding. Using this medical payments.
unified coding of diseases is conducive to storage, retrieval, and analysis In the early stages, ICD coding relied on manual work by professional
of medical data. ICD is also an effective means of standardizing medical coders. After browsing the available documentation, such as electronic
data, as well as forming the data basis for intelligent medicine applica- medical records (EMRs), the professional coder would manually select
tions. one or multiple suitable ICD codes from the tens of thousands of codes
ICD taxonomy is also the basis of the diagnosis-related groups (DRGs) that exist and assign them to the medical record. This process is error-
payment system. Wrong coding, omissions, or multiple coding will di- prone and time-consuming. Moreover, it requires coders to have good
rectly affect the quality of the data used for DRGs; the loss caused by medical knowledge and to be familiar with coding specifications and
such errors in ICD code assignment cannot be underestimated. Accord- rules. It is also costly, as professional coders need to be trained regularly
ing to statistics from the US Centers for Medicare and Medicaid, the error to keep up with the continuous updates of the ICD. From ICD-9-CM to
payout rate in 2000 was 6.8%; these errors were mainly caused by incor- ICD-10-CM, the number of ICD codes increased more than five-fold [3] .
∗ Corresponding authors: Xiangling Fu, School of Computer Science (National Demonstrative Software School), Beijing University of Posts and Telecommunications,
Beijing 100876, China (Email: fuxiangling@bupt.edu.cn ); Xien Liu, Department of Electronic Engineering, Tsinghua University, Beijing 100084, China (Email:
xeliu@mail.tsinghua.edu.cn ).
https://doi.org/10.1016/j.imed.2022.03.003
Received 15 November 2021; Received in revised form 5 January 2022; Accepted 11 March 2022
2667-1026/©2022 The Authors. Published by Elsevier B.V. on behalf of Chinese Medical Association. This is an open access article under the CC BY-NC-ND license
( http://creativecommons.org/licenses/by-nc-nd/4.0/ )

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
Figure 1. Developmental history of the automated ICD coding task.
The newest, and the latter version of the ICD codes can classify patients’ is “Intestinal invasive. ” Thus, they are unlikely to be assigned to the
conditions, injuries, and diseases in more detail than previous versions, same EMR at the same time. Furthermore, certain codes under different
resulting in substantial increases in both the number of codes and the parent nodes often appear in the same EMR owing to the complicated
difficulty of manual coding. internal logical relationship behind the disease (such as cold and cough);
To this end, researchers have proposed various automated coding we call these nodes friend nodes. They seems to have no connection in
approaches. The earliest research on this topic [4–5] , to the best of our the tree-like structure, but they often appear in pairs in EMRs.
knowledge, took place in the 1990s. We review in detail the progress Overall, these characteristics of parent–child nodes, sibling nodes,
made in this task in Section 4 . Figure 1 provides an overview of the his- and friend nodes in the ICD taxonomy can be classified into categories
tory of automated ICD coding research, showing the three development of inheritance, mutual exclusion, and co-occurrence. A detailed analysis
stages, milestones such as critical research or data publications, and how is presented in the following parts.
our understanding of the task has changed. Detailed task definitions will
be introduced in Section 3 . 2.1. Parent-child nodes: inheritance
Despite many advances in academic research, this task still faces se-
vere challenges in clinical applications. The first challenge is the large ICD codes are organized in a hierarchical structure. The parent codes
label space. Regarding a code as a label, the number of labels can reach are often a general disease category, and the child codes are refinements
tens of thousands. This huge search space makes both retrieval and clas- and supplements of their parent codes with respect to a certain disease
sification very difficult. The second challenge is the extremely unbal- feature. For example, the category code “E11 ”in ICD-10 represents type
anced label distribution [6] . A few ICD codes occur frequently in health- 2 diabetes; its child code “E11.3 ” means type 2 diabetes with ocular
related documents, whereas most codes are uncommon. The third chal- complications, which is a supplement to the parent code in terms of com-
lenge is the long document representation. The key is to extract useful plications; and the code “E11.302 ”represents type 2 diabetic cataract,
text segments to assist a models to map the diagnosis to ICD codes. Last which is a specific disease of ocular complications. Therefore, although
but not least, the interpretability of coding represents a fourth challenge. we are only required to predict a specific disease when we predict an
It is essential to provide a reasonable explanation for predicted codes in ICD code assign child ICD codes when we predict concrete diseases, the
actual applications. These challenges limit the practical applications of parent node also contains very relevant and useful information that can
coding tasks in clinical practice. give an approximate direction.
2.2. Sibling nodes: mutual exclusion
2. Characteristics of ICD codes
Sibling nodes in the same layer of the ICD taxonomy are classified by
The ICD taxonomy organizes codes in a hierarchical architecture their axis, which includes etiology, pathology, anatomical location, and
consisting of chapters, sections, and categories, even expansion codes. clinical manifestations. Most categories and sub-categories have only
Figure 2 illustrates an example of Escherichia coli ( E. coli ) infection in one axis; there are two classification axes in a few cases. This means
the ICD-11-MMS system. Assuming that every ICD code is a node, the that for a given layer, codes divided classified by the same axis are often
section codes and their corresponding chapter codes can be treated as mutually exclusive. According to a survey [7] , the third-ranking cause
child–parent relations. Thus, a node at one level may be the parent of of coding defects is logical conflicts of the classification axis, accounting
a node at the next level. Nodes that have the same parent node can be for 9.6% of all ICD coding defects. Specific logical conflicts include codes
regarded as sibling nodes. Clearly, child nodes are semantically similar representing a condition with with and without complications being as-
to their parents. It can be inferred that child nodes inherit some infor- signed to one medical record, or traumatic and non-traumatic codes for
mation from their parent node, whereas sibling nodes at the same level one disease being assigned at the same time.
represent refinements and extensions of the parent node with respect to
different aspects. Therefore, the difference between sibling nodes tends 2.3. Friend nodes: co-occurrence
to be mutually exclusive. Taking the E. coli infection in Figure 2 as an ex-
ample, “1A03.1 ”and “1A03.2 ”are sibling nodes, and both belong to the From the perspectives of disease complications and etiology, many
diagnosis of “E . coli infection ”. One is “Enterotoxigenic ”and the other diseases are related, and this relationship may even be causal. Patients
162

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
Figure 2. An example of E. coli infection in the ICD-11-MMS system.
diagnosed with disease A often suffer from disease B or disease C. For ex- input (diagnostic terms) and the output (codes) are two different lan-
ample, in the medical records of patients with acute myocardial infarc- guages, and that sequence-to-sequence models are suitable for the trans-
tion, heart failure (a common complication), coronary atherosclerosis lation from one to another. Similar to semantic similarity estimation,
(cause), or other related diseases often appear as secondary diagnosis. only the short text of the diagnosis is used as the input, rather than the
The typical representative of this phenomenon is co-morbidity recogni- whole document.
tion [8–10] . The correlations between types of disease are also reflected Multimodal Machine Learning. Benefiting from the abundance of
in the process of ICD coding; this is the co-occurrence aspect of coding. data, some studies [11,16,22] have made full use of structured and semi-
These three relations between ICD codes constitute the basic char- structured data, such as demographic data or laboratory results. Such
acteristics of ICD taxonomy. They enable the codes to be relevant and methods are often limited by the availability of datasets, but they rep-
meaningful, that is, they are no longer independent labels. This distin- resent a promising direction.
guishes the ICD coding task from general multi-label tasks. Multi-label Classification. In recent years, automated ICD coding
has increasingly often been treated as a multi-label classification task
3. Task definition and evaluation metrics [23–25] . The whole document is used as the input to obtain a text rep-
resentation, each ICD code is regarded as a label, and all the labels are
3.1. Task definition predicted simultaneously. The document type also extends to death cer-
tifications [26] , non-technical summaries (NTS) [27] , etc.
From the 1990s to the present, as our understanding of ICD coding There are two distinct differences in ICD coding multi-class text clas-
has evolved, this problem has been treated as different types of task. sification compared with general multi-class text classification. First, the
Thus, we first summarize how researchers view the task of ICD coding labels of the automated ICD coding task are organized in a hierarchical
and how they transform it into a learnable problem from different per- structure. As summarized in Section 2 , ICD codes exhibit the character-
spectives. istics of inheritance, mutual exclusion, and co-occurrence. Second, as
Information Retrieval. Initially, most studies treated automated medical text is knowledge-intensive, text classification models used in
ICD coding as an information retrieval task, using string matching, the general domain may not be able to understand the medical seman-
statistical processing, part-of-speech tagging, negative recognition, and tics used in health-related documents.
even medical language processing tools such as MedLEE to identify med-
ical concepts from structured and unstructured text [11] , so that the 3.2. Evaluation metrics
diagnosis could be matched with ICD codes [12–14] . Most documents
used in these studies were from EMRs. Many evaluation metrics have been proposed to evaluate the effec-
Binary Classification. With the rise of machine learning, many tiveness of the methods from different perspectives. The main evaluation
scholars began to regard the problem as a binary classification task. metrics are precision, recall, and F1, and each is calculated by micro-
They trained an independent classifier for each ICD code to determine average and macro-average approaches.
whether a code should be assigned to the corresponding document, and The area under the receiver operating characteristic curve (AUC) and
then integrated the prediction results of all the classifiers as the final 𝑃 @𝑘 (precision at 𝑘 , the fraction of the 𝑘 highest-scoring labels that
prediction result [15-16] . This classification method is often used in sit- are present in the ground truth) are also commonly used. For decision
uations where there are not many ICD codes, such as radiology reports. support, it is convenient to show a fixed number of predictive codes to
Semantic Similarity Estimation. In addition, some studies [17- users. The value of 𝑘 is usually selected as 5, 8, or 15 [23,28-29] .
18] have defined ICD coding as a semantic similarity estimation task. Macro-average metrics are calculated by averaging the metrics cal-
Such approaches only utilize the short text describing the actual diag- culated for each label. Owing to the multiple and unbalanced classes
nosis (the remainder of the document is discarded) and compare the di- involved in this task, macro-average metrics place more emphasis on
agnosis with the disease description in the ICD coding taxonomy [19] . the prediction of rare labels, which can better reflect the performance
The common computation way between text and label is cosine simi- of the model in small classes. The micro average-recall rate and macro
larity [20] . Such methods require high accuracy and standardization of average recall rate are calculated as follows:
the
M
do
a
c
c
t
h
o
i
r
n
’s
e
w
T
r
r
i
a
tt
n
e
s
n
l a
d
t
i
i
a
o
g
n
n
.
o
M
sis
o
.
reover, the task may be viewed as a spe- 𝑀𝑖𝑐𝑟𝑜 − 𝑅 = ∑
∑
| 𝑙 𝐿 =1 | 𝑇 𝑃 𝑙 ) , (1)
cial type of machine translation problem [21] . It is assumed that the | 𝑙 𝐿 =1 | 𝑇 𝑃 𝑙 + 𝐹 𝑁 𝑙
163

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
conflicts to arise among different rules. This problem can be manually
𝑀𝑎𝑐𝑟𝑜 − 𝑅 = 1 ∑|𝐿 | 𝑇 𝑃 𝑙 , (2) screened and adjusted for in the CCHMC dataset as it only contains 45
|𝐿 | 𝑙=1 𝑇 𝑃 𝑙 + 𝐹 𝑁 𝑙 ICD codes. However, if the number of codes rises to thousands or tens
of thousands, rule-based methods may no longer be feasible.
where TP stands for true positive cases and FN stands for false negative
cases. The calculation of accuracy is similar. 4.2. Stage 2: traditional machine learning-based methods
3.3. Summary With the emergence of machine learning, many studies began to use
machine-learning-based methods to tackle the automated ICD coding
In summary, automated ICD coding is the process of assigning ICD task (Table 1). The key to traditional machine-learning-based methods
codes 𝑦 to health-related documents 𝐷. These documents commonly in- is the construction and selection of features [35] .
clude EMRs, death certifications, clinical summaries, and medical re- Suominen et al. [36] used feature engineering methods to extract
ports. features such as word segmentation results, medical concepts, and hy-
More specifically, ICD coding requires the mapping of a series of pernyms, and then used a cascade of two classifiers: the regularized least
diagnoses 𝐷𝑖𝑎𝑔 = [ 𝑑 𝑖𝑎𝑔 1 , 𝑑 𝑖𝑎𝑔 2 , …, 𝑑𝑖𝑎𝑔 𝑘 ] in document 𝐷 to a series of squares (RLS) classifier and the RIPPER algorithm, to determine the fi-
codes 𝑦 = [ 𝑖𝑐 𝑑 1 , 𝑖𝑐 𝑑 2 , …, 𝑖𝑐𝑑 𝑘 ] . Through representation learning, the doc- nal multi-label predictions. In experiments conducted on the CCHMC
ument 𝐷is represented as a sequence of surrounding segments of diag- dataset, they achieved an 87.7% micro-averaged F1-score. Perotte et al.
nosis 𝑆 𝑑𝑖𝑎𝑔 𝑗 , where 1 ≤ 𝑗 ≤ 𝑘 . 𝑆 𝑑𝑖𝑎𝑔 𝑗 can be narrowly understood as the [32] took keywords obtained from Term Frequency-Inverse Document
disease name or a short description of the diagnosis result. In a broad Frequency (TF-IDF) as features and proposed two models, FlatSVM and
sense, it is the set of all segments represented in the document that are hierarchy-based SVM. The FlatSVM model treats each prediction inde-
related to 𝑑𝑖𝑎𝑔 𝑗 . Take EMRs as an example: the direct diagnosis appears pendently. Marafino et al. [37] extracted n-gram features to build a bi-
in the discharge diagnosis; however, the admission situation, admission nary SVM classifier for each disease. Elyne et al. [16] took both struc-
diagnosis, discharge situation, examination results, and other segments tured and unstructured clinical data into consideration, and a combined
may also be related to the diagnosis and can be used as surrounding bag of words (BoW) from the unstructured text was selected as one of
information to assist coding. the features.
Thus, the core process of this problem can be abstracted as the repre- Most traditional machine learning methods train a separate classifier
sentation learning of documents to obtain diagnosis-related information for each code. These methods can achieve good results when the num-
and the classification of ICD codes from the diagnosis. ber of codes to be predicted is small. For example, the CCHMC dataset
used by Suominen et al. [36] contained 45 codes, and the experiment of
4. Developmental history of automated ICD coding Marafino et al. [37] was only conducted on four diseases, and positive
and negative training samples were manually balanced. However, the
The development of automated ICD coding has taken place over number of ICD codes is rapidly increasing, and it is impractical to train
decades. We divide it into three stages based on the techniques used. (1) thousands of classifiers in this way. Another drawback is that each dis-
In the rule-based stage, methods tried to extract and transfer the rules ease is considered independently and the relationships between multiple
of coding specification to if-else logical programs to replace repetitive ICD codes assigned to one EMR are ignored.
manual work [30] . After that, (2) traditional machine learning meth-
ods, such as support vector machine (SVM), were used to predict ICD 4.3. Stage 3: neural-network-based methods (Table 2)
codes from health-related documents [31–33] . At this stage, the focus
was on constructing distinguishable features and finding suitable classi- With the development of deep learning, studies of automated ICD
fiers. With the development of neural networks, (3) deep learning meth- coding based on neural networks have gradually become mainstream.
ods emerged to tackle the ICD coding prediction, and the focus shifted CNN-based methods. The most classic examples are the CAML and
to how to obtain better representations of health-related documents by DR-CAML models proposed by Mullenbach et al. [23] . These models use
designing different neural networks. In this section, we review some of convolutional neural networks (CNNs) to automatically extract features
these methods and summarize their performance on the publicly avail- in the discharge summaries, followed by a per-label attention mecha-
able Medical Information Mart for Intensive Care (MIMIC) dataset. nism. To obtain the probability of each label, sigmoid transformation is
used for multi-label prediction. The models are simple but efficient and
4.1. Stage 1: rule-based methods robust; they demonstrated breakthrough improvements on the MIMIC-II
and MIMIC-III datasets. The micro-average F1 reached 0.457 (MIMIC-
Early automated ICD coding methods mainly relied on rules and II), 0.633 (MIMIC-III-50), and 0.529 (MIMIC-III-Full).
expert experience [30,34] to mine and automatically generate coding Many subsequent studies were inspired by and improved on these
rules by learning ICD guidelines. The ICD coding guidelines indicate the models. Cao et al. [38] and Ji et al. [39] used a dilated CNN, which
symptoms, signs, and corresponding constraints for each disease. Thus, inserted “holes ”into the filters, to extract non-continuous semantic in-
many methods tried to convert the plain text into executable logical formation from the EMRs. Some researchers have argued that a flat and
judgments so as to achieve automatic coding prediction. In addition, ex- fixed-length convolutional architecture may not be able to learn a good
panding the list of medical concepts in the coding guidelines was also an document representation. Thus, multi-scale and variable-sized convo-
important way to enhance the effectiveness of rule-based coding meth- lutional filters have been employed to capture various text patterns of
ods. This could increase the concept coverage of health-related by us- different lengths [40–43] .
ing corresponding concepts in the guidelines with their abbreviations, These methods try to expand the receptive fields of filters and match
synonyms, etc. According to the work of Farkas and Szarvas [30] , the different text patterns by changing the shape, structure, or scale of the
performance of rule-based methods on the Cincinnati Children’s Hospi- filters, with the aims of obtaining a better text representation. Examples
tal Medical Center (CCHMC) dataset could reach 90.26% (training set) of these filters are shown in Figure 3 . As well as multi-scale convolu-
and 88.93% (test set). tion, Li and Yu [42] and Ji et al. [39] added a residual block [44] on
However, the disadvantages of the rule-based methods are their poor the convolution layers; this was also beneficial for the expansion of the
flexibility and portability. There are many common symptoms among receptive field.
different diseases; this can easily lead to over-coding and missing-code Recurrent neural network (RNN)-based methods. In another se-
problems. More seriously, as the number of codes increases, it is easy for ries of studies, RNNs were employed to extract features from medical
164

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
an information retrieval system was used to obtain the final prediction.
Wang et al. [54] constructed a heterogeneous graph from each EMR,
enriched by the Wikipedia database. They pre-trained a graph encoder
with two graph contrastive learning schemes.
Pre-trained language model (PLM)-based methods. In recent
years, PLMs have achieved state-of-the-art results on multiple NLP natu-
ral language processing tasks. A few studies have employed PLMs to en-
hance the ability of text representation. Owing to the occurrence of spe-
cialized terms and expressions, BERT [55] is not well-suited to medical
texts; however, the pre-training models specifically trained on biomed-
ical texts, such as BioBERT [56] , ClinicalBERT [57] , and PubMedBERT
[58] , may better help to understand such texts.
Zhang et al. [59] trained BERT on EMRs and used it to encode text.
They extended the maximum sequence length to 1,024; 90% of docu-
ments in their dataset did not contain more than 800 tokens, and the
Figure 3. The different filters used in CNN-based methods.
documents that exceeded 1,024 tokens were split into segments. The
highest predicted probability per ICD code across segments was used
as the note-level prediction. As shown in Table 4 , the average token
texts. Shi et al. [45] used two-level long short-term memory (LSTM) net- number per document in common EMR datasets is no less than 1,100,
works (character-level and word-level) to obtain representations of the and the maximum token number of the MIMIC dataset exceeds 2,500.
diagnosis description and ICD title, respectively. Baumel et al. [46] em- Therefore, splitting input text into segments is inevitable. Pascual et al.
ployed two layers of Gated Recurrent Unit (GRU) to encode the doc- [60] directly employed PubMedBERT [47] as an encoder and explored
uments. The first layer encoded tokens to generate sentence repre- different ways to split text. A decoder was also needed to combine the
sentations, and the second layer encoded the sentences to obtain a representations of each chunk. Surprisingly, the experiments illustrated
whole-document representation. The attention mechanism was applied that splitting the text into chunks of the same size resulted in the worst
to both layers. Yu et al. [24] proposed a multi-layer attention bidi- performance; even taking only the first 512 tokens or last 512 tokens
rectional RNN (MA-BiRNN) model, in which one layer learned word- was a better approach.
level features of a document, and the other layer learned word-level Based on the results obtained with the above PLM-based methods,
features. although PLMs show outstanding ability in understanding medical lan-
It is challenging to model long sequences and realize parallel com- guage, they do not perform well on ICD coding tasks. This may ac-
puting with RNNs; thus, the above works used multi-layer structures to count for few PLM-based methods being developed for this task. The
decompose the documents and reduce the length of the sequence. Yu poor performance may be affected by the long length of input medi-
et al. [24] proposed, the average number of words per document was cal documents. PLMs can often only process 512 tokens at once, which
610, and Shi et al. [45] only selected the diagnosis description from the is less than the token number of most documents used in this task,
discharge summary to encode. Another approach to decrease the length and the necessary splitting of text has a negative impact on the final
of the document is to extract the symptoms from the text and use LSTM performance.
to model the relation between symptom sequence and disease [47] . It
can be inferred that a short text length is vital to the performance of 4.4. Summary
RNN-based models.
Graph neural network (GNN)-based methods. GNNs have also In summary, the automated ICD coding task has been a hot research
been applied in automatic coding. The main method is to employ GNNs topic in recent years, and many excellent methods and models have
to learn the representation of ICD codes, as the codes are organized in been proposed. In addition, many new sub-tasks and processing methods
a tree-based structure. Rios and Kavuluru [48] used the graph convo- have been developed to assist automated ICD coding tasks, such as the
lutional network (GCN) model to obtain features for each label, aiming prediction of the number of labels [15] .
to solve the few-shot and zero-shot prediction problems in ICD coding. To obtain a more intuitive understanding of the progress that has
Similarly, Du et al. [49] used the tree-based structure of ICD taxonomy been made, we summarize the results of state-of-the-art methods on the
to capture the dependency of codes. Cao et al. [25] also counted the MIMIC datasets in Table 3 , as MIMIC is the most widely used public
co-occurrence frequency of ICD codes in EMRs to learn a code represen- dataset. To be specific, there are three datasets: MIMIC-II, MIMIC-III-
tation by GCN; this improved the incomplete prediction of coding. More- 50, and MIMIC-III-Full.
over, Wang et al. [50] proposed two approaches to edge construction in In the first block of the table, the performances of three typical mod-
the ICD code graph: point-wise mutual information between codes, and els are reported, as implemented by Mullenbach et al. [23] . These text
the TF-IDF value. Although they did not conduct experiments on the classification methods did not show satisfactory performance in general,
ICD coding task, their work provides a reference for graph construction. and there was a significant gap in the F1 scores of these typical mod-
Gao et al. [51] proposed an unsupervised semantic-based heterogeneous els and those obtained with improved deep learning models. The best
graph representation method (SMP-Graph) that inductively enhanced Micro-F1 scores on MIMIC-II, MIMIC-III-50, MIMIC-III-Full were 0.498,
the axis word-level, chapter-level, and document-level knowledge im- 0.725, and 0.575, respectively. The Fusion method proposed by Luo
plicit in each piece of coding text. et al. [61] , ISD proposed by Zhou et al. [62] , LAAT&JointLAAT proposed
A few studies have used knowledge graphs (KGs) to enhance the rep- by Vu et al. [63] , and MSATT-KG proposed by Xie et al. [64] showed
resentation of medical texts or ICD codes. Teng et al. [52] built a KG with the most competitive performance.
more than 1,500 nodes by fusing ICD-9 descriptions and medical-related It can also be inferred from the results shown in Table 3 that the
data from the Freebase database. THe SDNE algorithm was used to en- medical field is knowledge-intensive, and it is more difficult to cap-
code the ICD codes, and the KG embedding helped to understand the ter- ture features from medical text than general texts, hence the need to
minology. By contrast, Chelladurai et al. [53] constructed an initial con- design more elaborate network structures or use more external knowl-
textual graph with entities extracted from the clinical notes and those edge. Most of the methods listed in Table 3 have been introduced
enriched by a pre-constructed external KG. Then, a GNN was employed in stages 2 and 3 above; the others will be described in detail in
to filter relevant nodes in the graph and contextualize the concepts, and Section 6 .
165

C. Yan, X. Fu, X. Liu et al.  Intelligent Medicine 2 (2022) 161–173
Table 1 Traditional machine-learning-based methods
| researches                | Models                                                   |     | Methods/Features   |     |
| ------------------------- | -------------------------------------------------------- | --- | ------------------ | --- |
| (Larkey&Croft, 1996) [5]  | K-nearest-neighbor; Bayes classifier;relevance feedback  |     | Terms and phrases  |     |
(Suominen et al., 2007) [36]  RLS and RIPPER  Word segmentation,medical concepts, and hypernyms
| (Perotte et al., 2013) [32]   | FlatSVM and hierarchy-based SVM  |     | TF-IDF  |     |
| ----------------------------- | -------------------------------- | --- | ------- | --- |
| (Marafino et al., 2014) [37]  | SVM                              |     | N-gram  |     |
(Elyne et al., 2016) [16]  Naive Bayes classifier and random forests  BoW;TF-IDF;demographic data;laboratory results, etc.
Table 2 Neural network-based methods
Input medical text (all
text in the document
| Approaches  | Model descriptions  |     | included?)  | Output layer  |
| ----------- | ------------------- | --- | ----------- | ------------- |
CNN-based
√
CAML & DR-CAML [23]  CNN + label-wise attention  √ Sigmoid activation
DCANM [38]  Dilated CNN + N-gram matching mechanism  Sigmoid activation
√
MVC-LDA & MVC-RLDA [40]  Max pooling across the  Sigmoid activation
|     |     | channels  | √   |     |
| --- | --- | --------- | --- | --- |
Ontological attention [41]  Ontological attention  Sigmoid activation
|     | Multi-ScaleCNN  | and mapping  |     |     |
| --- | --------------- | ------------ | --- | --- |
√
MultiResCNN [42]  Residual convolutional  Sigmoid activation
|              |     | layer                | √   |                      |
| ------------ | --- | -------------------- | --- | -------------------- |
| EnCAML [43]  |     | Concatenate feature  |     | Sigmoid              |
|              |     | maps horizontally    |     | activation(variable  |
threshold)
RNN-based
C-LSTM-ATT [45]  English letter-level +  Diagnosis descriptions  Sigmoid activation
|     | LSTM  | word-level  |     |     |
| --- | ----- | ----------- | --- | --- |
MA-BiRNN [24]  Chinese character-level +  Feature words  Sigmoid activation
|     |     | word-level  | √   |     |
| --- | --- | ----------- | --- | --- |
HA-GRU [46]  GRU  Token-level +  (Split into sentences)  Sigmoid activation
sentence-level
|     | GNN-based  |     | √   |     |
| --- | ---------- | --- | --- | --- |
ZAGCNN [48]  GCNs is employed to  Use both label  Sigmoid activation
|     | learn code representation  | descriptors and structure  |     |     |
| --- | -------------------------- | -------------------------- | --- | --- |
√
| HyperCore [25]  |     | Graph constructed by  |     | Sigmoid acivation  |
| --------------- | --- | --------------------- | --- | ------------------ |
codes co-occurrence
√
| MSResGCN [49]  |     | Label graph: ICD  |     | Sigmoid activation  |
| -------------- | --- | ----------------- | --- | ------------------- |
tree-based structure
√
GrabQC [53]  GNN is to encode  Recognize named  Information retrieval
|     | documents   | entities and link to KG  |     | system  |
| --- | ----------- | ------------------------ | --- | ------- |
|     | PLMs-based  |                          | √   |         |
BERT-XML [59]  Train BERT model on EMRs; most documents do not  Sigmoid activation
need to be split as the maximum sequence length is
extended to 1,024
√
BERT-ICD [60]  PubMedBERT; five text-splitting strategies  Not mentioned
5. Commonly used datasets for automated ICD coding  The remainder of the datasets are provided by various hospitals and
medical institutions. UKLarge and UKSmall [15] collected EMRs from
Many datasets in various languages have been exploited and made  the University of Kentucky Medical Center between 2011 and 2012. The
public. We list 17 commonly used datasets in Table 4 and report their de-  Xiangya dataset [24] is derived from three affiliated hospitals of Central
tailed statistical information. Similar to most previous studies, we only  South University. The CN-Full dataset [38] is derived from a coopera-
performed statistical analyses on the unstructured text information in  tive medical institution. The EMRs of UZA [16] are from the Antwerp
the MIMIC dataset.  University Hospital.
The most commonly used dataset is MIMIC [66–67] . It collects com-  Conference and Labs of the Evaluation Forum (CLEF) has evaluated
prehensive clinical data from tens of thousands of intensive care unit  ICD coding of health-related documents every year from 2017 to 2020,
(ICU) patients from 2001. The data include discharge summaries, radi-  and its datasets are mainly based on French, Spanish, German, and other
ology reports, laboratory measurements, microbiology cultures, medica-  European languages. The CDC (Center for Disease Control) dataset and
tion prescriptions, vital signs, and other numerical and textual data. The  CepiDc-2017 dataset [69] are the specific datasets used in CLEF eHealth
diagnoses in the MIMIC dataset are all labeled by ICD-9 codes. MIMIC-  2017. The CepiDc-2018, CLEF-Italian, and CLEF-Hungarian datasets
III-50 is a subset of MIMIC-III-Full, including the EMRs labeled by at  [26] were publicly available for use in CLEF eHealth 2018. They all
least one of the top 50 most frequently used codes. MIMIC-IV [67] is  collect electronic death certificates as the documents to be assigned ICD
the latest version of the MIMIC datasets. Unlike MIMIC-III, MIMIC-IV is  codes.
grouped into several modules, i.e., Core, Hosp, ICU, ED, CXR, and Note.  THe CLEF-German dataset [27] , released by CLEF in 2019, consists
The note module, which contains free-text clinical notes, has not been  of NTS of animal experiments. Each NTS contains a title, uses (goals) of
made publicly available.  the experiments, possible harms caused to the animals, and comments
The CCHMC dataset [68] , also called the Computational Medicine  about replacement, reduction, and refinement [70] . It is annotated with
Center dataset, was widely used in the early period of research on this  chapters or group codes from the ICD-10 German Modification 2016
topic and collects radiology reports from the CCHMC. It exploits the  version. For example, the entorhinal incoming of epilepsy the Entorhinal
“majority ”rule, and each document is labeled by only one code.  Afferents in Epilepsy experiment carried out on mice is labeled with VI
166

C.
Yan,
X.
Fu,
X.
Liu
et
al.
Intelligent
Medicine
2
(2022)
161–173
Table 3 The results reported on the MIMIC-II, MIMIC-III-Full, and MIMIC-III-50 datasets
Models MIMIC-II MIMIC-III-50 MIMIC-III-Full
AUC F1 AUC F1 AUC F1
P @8 P @5 P @8 P @15
Mac Mic Mac Mic Mac Mic Mac Mic Mac Mic Mac Mic
Logistic regression [23] 0.690 0.934 0.025 0.314 0.425 0.829 0.864 0.477 0.533 0.546 0.561 0.937 0.011 0.272 0.542 0.411
CNN [23] 0.742 0.941 0.030 0.332 0.388 0.876 0.907 0.576 0.625 0.620 0.806 0.969 0.042 0.419 0.581 0.443
Bi-GRU [23] 0.780 0.954 0.024 0.359 0.420 0.828 0.868 0.484 0.549 0.591 0.822 0.971 0.038 0.417 0.585 0.445
Flat SVM [32] - - - 0.211 - - - - - - - - - - - -
Hierarchy-based SVM [32] - - - 0.293 - - - - - - - - - - - -
C-LSTM-ATT [45] - - - - - - 0.900 - 0.532 - - - - - - -
C-MemNN [28] - - - - - 0.833 - - - 0.420 - - - - - -
HA-GRU [46] - - - 0.366 - - - - 0.366 - - - - - - -
CAML [23] 0.820 0.966 0.048 0.442 0.523 0.875 0.909 0.532 0.614 0.609 0.895 0.986 0.088 0.539 0.709 0.561
DR-CAML [23] 0.826 0.966 0.049 0.457 0.515 0.884 0.916 0.576 0.633 0.618 0.897 0.985 0.086 0.529 0.690 0.548
LEAM [20] - - - - - 0.881 0.912 0.540 0.619 0.612 - - - - - -
MA-BiRNN [24] - - - - - - - - - - - - - 0.420 - -
MSATT-KG [64] - - - - - 0.914 0.936 0.638 0.684 0.644 0.910 0.992 0.090 0.553 0.728 0.581
KAICD [65] - - - - - - - - - - - - - 0.462 - -
HyperCore [25] 0.885 0.971 0.070 0.477 0.537 0.895 0.929 0.609 0.663 0.632 0.930 0.989 0.090 0.551 0.722 0.579
DACNM [38] - - - - - 0.890 0.916 0.579 0.641 0.616 - - - - - -
DCAN [39] - - - - - 0.902 0.931 0.615 0.671 0.642 - - - - - -
MSResGCN [49] - - - - - - - - - - 0.877 0.983 0.076 0.539 0.722 0.568
MultiResCNN [42] 0.850 0.968 0.052 0.464 0.544 0.899 0.928 0.606 0.670 0.641 0.910 0.986 0.085 0.552 0.734 0.584
LAAT [63] 0.868 0.973 0.059 0.486 0.550 0.925 0.946 0.666 0.715 0.675 0.919 0.988 0.099 0.575 0.738 0.591
JointLAAT [63] 0.871 0.972 0.068 0.491 0.551 0.925 0.946 0.661 0.716 0.671 0.921 0.988 0.107 0.575 0.735 0.590
BERT-ICD [60] - - - - - 0.845 0.887 - - - - - - - - -
G_coder [52] - - - - - - 0.933 - 0.692 0.653 - - - - - -
ISD [62] 0.901 0.977 0.101 0.498 0.564 0.935 0.949 0.679 0.717 0.682 0.938 0.990 0.119 0.559 0.745 -
Fusion [61] - - - - - 0.931 0.950 0.683 0.725 0.679 0.915 0.987 0.083 0.554 0.736 -
The best results are underlined, and those that are less than 0.05 from the best result are marked in bold. This table only compares methods that use the same text pre-processing technique; thus, the results of
experiments conducted on the MIMIC dataset using different pre-processing techniques are not listed.
167

C. Yan, X. Fu, X. Liu et al.
Table 4 Datasets commonly used for ICD coding
Language  Dataset  Public  Data Format  Document Type  ICD Version  # Documents  Avg Token # per  Avg Labels# per  Total # Labels
|     |     |     |     |     |     | Document  | Document  |     |
| --- | --- | --- | --- | --- | --- | --------- | --------- | --- |
√
English  MIMIC-II  ○1 ○2  EMRs  ICD-9  20,533  1,138  9.2  5,031
√
MIMIC-III-Full  ○1 ○2  EMRs  ICD-9  47,724  1,485  15.9  8,922
√
|     | MIMIC-III-50  | √ ○1 ○2  | EMRs               | ICD-9     | 8,067  | 1,530  | 5.7  | 50  |
| --- | ------------- | -------- | ------------------ | --------- | ------ | ------ | ---- | --- |
|     | CCHMC         | ○2       | Radiology reports  | ICD-9-CM  | 1,954  | 21     | 1    | 45  |
√
CDC  ○2  Death certificate  ICD-10  13,330 (train)  6.8  3.0  1,256
|     |          |        |       |           | 6,665 (test)  | 6.4    | 2.8  | 900    |
| --- | -------- | ------ | ----- | --------- | ------------- | ------ | ---- | ------ |
|     | UKLarge  | ✗  ○2  | EMRs  | ICD-9-CM  | 71,463        | 5,303  | -    | 1,231  |
|     | UKSmall  | ✗  ○2  | EMRs  | ICD-9-CM  | 1,000         | 2,088  | -    | 56     |
Chinese  Xiangya  ✗  ○2  EMRs  ICD-10  7,732  610  3.6  1,177
|     | CN-Full  | ✗  ○2  | EMRs  | ICD-10  | 50,678  | 621  | 4.3  | 6,200  |
| --- | -------- | ------ | ----- | ------- | ------- | ---- | ---- | ------ |
|     | CN-50    | ✗  ○2  | EMRs  | ICD-10  | 36,758  | 655  | 2.6  | 50     |
168
| Dutch  | UZA  | ✗  ○1 ○2  | EMRs  | ICD-9-CM  | 56,641  | -   | -   | 23,727  |
| ------ | ---- | --------- | ----- | --------- | ------- | --- | --- | ------- |
√
French  CepiDc-2017  ○2  Death certificates  ICD-10  65,844 (train)  17.9  4.1  3,233
|     |     |     |     |     | 27,850 (test)  | 17.8  | 4.0  | 2,363  |
| --- | --- | --- | --- | --- | -------------- | ----- | ---- | ------ |
√
CepiDc-2018  ○2  Death certificates  ICD-10  125,384 (train)  -  -  -
11,932 (test)
√
Italian  CLEF-Italian  ○2  Death certificates  ICD-10  14,502 (train)  -  -  -
3,618 (test)
√
Hungarian  CLEF-Hungarian  ○2  Death certificates  ICD-10  84,703 (train)  -  -  -
|     |     | √   |     |     | 21,176 (test)  |     |     |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- | --- |
German  CLEF-German  ○2  NTS of animal experiments  ICD-10 German modification  8,386 (train & dev)  369.17  2.6  233
|     |     |     |     | 2016 version  | 407 (test)  |     |     |     |
| --- | --- | --- | --- | ------------- | ----------- | --- | --- | --- |
√
Spanish  CLEF-Spanish  ○2  EMRs  Spanish version of ICD10-CM  1,000  396.99  18.4  3,427
and ICD10-PCS
| ○ 1 :Structured and ○ | 2 Unstructured.  |     |     |     |     |     |     |     |
| --------------------- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
Intelligent Medicine 2 (2022) 161–173

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
and G40-G47, where VI is the code of the Nervous System Disease the Given the characteristic of inheritance of ICD codes, Vu et al.
Diseases of the Nervous System chapter and G40-G47 is the code of the [59] proposed the JointLAAT model to learn the hierarchical relation-
Transient and Sudden Illness Episodic and Paroxysmal Diseases of the ship between codes through a hierarchical joint learning mechanism.
Nervous System section. Falis et al. [40] proposed ontological attention to capturing semantic
The CLEF-Spanish dataset [71] , also called the CodiEsp dataset, is concepts in ICD code prediction from the clinical text by considering
the specific dataset used in CLEF eHealth 2020: 1,000 EMRs are used the hierarchical structure of the coding; three-layer coding is used as a
for the training set, development set, and test set. In addition, 2,751 label-wise approach to improve performance.
unlabeled EMRs are provided as a background set. The EMRs include Regarding the mutual exclusion of ICD codes, various methods exist
documents labeled by both diagnosis codes and procedure codes. Ignor- for preventing sibling nodes from being selected simultaneously. Sub-
ing the type of document, the average number of labels for 1,000 EMRs otin and Davis [72] considered simultaneous prediction as an asyn-
is 18.4, as reported in Table 4 . However, the average number of labels chronous process and introduced conditional probability to learn the
for each procedure document is 8.2, whereas the number for diagnosis probability of code B being assigned given that code A has been as-
documents is 25.27. signed to the medical record. Pengtao et al. [73] used the sequence
The majority of the datasets are in English, followed by Chinese. The tree-based LSTM network to capture the relationship between codes,
rest are in various European languages. effectively preventing a common choice of sibling codes. Moreover, Cao
Most of the English-language datasets are publicly available and used et al. [25] converted the ICD taxonomy from Euclidean space to hyper-
widely. Among them, the MIMIC dataset is the most frequently used bolic space, increasing the distance between nodes at the same level in
dataset in published studies. The European-language-based datasets are hyperbolic space to reflect the mutual exclusion relationship of sibling
public, whereas the Chinese-language datasets are all non-public. nodes.
The datasets consist of health-related documents, mainly EMRs, radi- The co-occurrence of coding has also been noted in recent years.
ology reports, death certificates, and NTS reports. The EMRs are mainly Cao et al. [25] calculated the frequency of co-occurrence of codes in
collected from hospitalized patients. The document lengths of the death EMRs and built a code graph where the edge weight represented the fre-
certificates and radio-graphic reports are relatively short, with an aver- quency. Graph convolutional networks have been utilized to learn the
age length of between 6.4 and 21 English words. The average length of co-occurrence relationship between codes and obtain a representation of
English EMRs ranges from 1,138 to 5,303 words, and the average length the code. In addition to considering the co-occurrence relationship dur-
of Chinese EMRs is about 600 words. ing modeling, it is also useful to consider it in the post-processing stage
Regarding the total number of ICD codes in the datasets, with the to improving the final result. Tsai et al. [74] regarded the automated
exception of the CCHMC dataset, which only contains 45 ICD codes, coding task as a two-stage task, where the first stage is model predic-
most of the datasets have no fewer than 1,000 ICD codes. The MIMIC- tion, and the second stage is re-ordering the predicted codes based on
III dataset contains the most codes, with a total number of 8,922. For the label distribution of the dataset. The whole process is equivalent
datasets with large numbers of tags, such as the MIMIC-III-Full dataset, to adding post-processing to the original model. Experiments show that
CN-Full dataset, and UKLarge dataset, some codes with a higher fre- adding constraints of the co-occurrence relationship to three existing
quency are selected to generate a corresponding subset: the MIMIC-III- models (CAML, MultiResCNN, and LAAT) can improve the effect of the
50, CN-50, and UKSmall datasets. original model.
6. Challenges and existing solutions in automated ICD coding
6.2. Unbalanced label distribution
Automated ICD coding is an important fundamental task in the do-
The extremely unbalanced distribution of codes is the second biggest
main of smart medicine. Although many studies have attempted to ad-
challenge in the task of automated ICD coding. A few codes appear very
dress this task using many different approaches, varying from rule-based
frequently in EMRs, such as respiratory infections and coughs, whereas
to neural-network-based methods, and achieved great improvements in
most of the codes have a very low frequency of appearance; this leads to
performance on several evaluation metrics, there are still many chal-
the serious long-tails phenomenon [6] . Some studies also refer to it as
lenges to be solved. These include the high dimension of the label space,
the Zipffe distribution of the ICD codes in EMRs, and the datasets with
severe imbalances in numbers of samples for each disease/code, irreg-
long tails are called “power-law datasets ”[75] .
ular language expressions, and inevitable noise in EMRs. These prob-
We investigated some datasets to explore the specific distribution
lems become more prominent in the practical clinical environment, and
of ICD codes; the results are shown in Figure 4 . In the MIMIC-III
there is a long way to go before a robust, reliable, and explainable AI
dataset, 10% of ICD codes appear in 85% of the data, whereas 22%
infrastructure is constructed. Thus, we summarize four challenges of au-
of codes appear fewer than two times, and about 5,000 labels appear
tomated ICD coding tasks and and also review some outstanding work
between one and ten times. More seriously, more than 50% of diagnos-
that attempts to break through the bottleneck.
tic and procedure codes, around 17,000, never appear in the dataset
[45,60] .
6.1. Large label space The situation with respect to the Chinese datasets is even more se-
rious. In CN-Full and CN-50 [38] , 32.8% of codes only appear once,
The large number of ICD codes leads to a huge label space for the cod- 46.9% of codes appear less than two times, and 74.6% of codes appear
ing prediction task. ICD-9-CM includes about 13,500 diagnostic codes between one and ten times. The most high-frequency code (Essential
and 4,000 procedure codes, each of which contains a maximum of four hypertension) appears 9,544 times in EMRs.
digits. In the next version, ICD-10-CM, the number of diagnostic codes The highly frequent occurrence of a minority of codes is consistent
increases to more than 70,000, the number of procedure codes increases with the situation in real medical settings. Most patients in a hospital
to 72,000, and the number of digits per the code reaches seven [3] . The suffer from common diseases; a few additional diseases may be encoun-
large label space leads to difficulties with coding prediction, and it is tered occasionally, and many rare diseases in the ICD taxonomy may
expected to become larger still with further iterations of the ICD. never appear. In addition, some ICD codes recently introduced in the
Most approaches to deal with this problem use the three characteris- latest version may be rarely or never used.
tics summarized in Section 2 , that is, they make use of the hierarchical This problem is likely to be missed by data-driven machine learn-
architecture of the ICD taxonomy, explore the relationships between ing methods because the wrong prediction of infrequent labels has lit-
codes, and conduct reasonable statistical analysis. tle effect on the final evaluation metrics. However, it is important for
169

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
Figure 4. The label frequency distribution in different datasets.
both medical records management and medical reimbursement for such tic code to allow different labels in the same high-level category to share
rare labels to be correctly predicted. The main deep-learning-based semantics in order to solve the problem of label imbalance. They took
methods intended to deal with this challenge can be summarized as the high-level coding prediction as the auxiliary task to train together
follows. with the low-level coding task. Zhou et al. [60] proposed a shared at-
Sampling. The general strategy to address the problem of unbal- tention representation, extracting the shared features of low-frequency
anced label distribution in machine learning is to improve it through coding and high-frequency coding, to overcome the difficulty of learning
methods such as random under- or over-sampling, synthetic training accurate representations of rare codes owing to insufficient data. Wang
sample generation, and cost-sensitive learning. Kavuluru et al. [15] con- et al. [78] used both the “Signs and symptoms ”section in Wikipedia and
structed an “optimal training set ”for each label by under-sampling neg- the ICD hierarchical structure.
ative samples. However, this method converts the multi-label classifica- Moreover, transfer learning can be used to transfer domain knowl-
tion problem into multiple binary classification problems, which do not edge. Zeng et al. [79] utilized automatic MeSH indexing as an extra
have universal adaptability. auxiliary task to learn domain knowledge and transferred it to an au-
Introducing the n ame and d escription of the d isease. The disease tomatic ICD coding task. MeSH is a medical literature data source that
name is the most intuitive piece of information. Shi et al. [43] encoded contains tens of millions of samples.
the disease name (long title) encoded by the ICD to generate a label
representation. Wu et al. [58] used two different networks to extract 6.3. Long text of documents
the features of disease names and the medical record text. The medical
record text is extracted using a multi-scale CNN, and the disease name Generally speaking, health-related documents may be very long; this
is extracted using Bi-GRU. For codes whose disease names themselves applies especially to EMRs, which include past history, current history,
do not provide enough information, a deeper understanding can be ob- examination results, diagnosis, etc., and use many complicated medi-
tained from the disease descriptions given in the ICD coding system. Rios cal expressions and professional terms. Moreover, many abbreviations,
and Kavuluru [45] used a GCN model to improve the few-shot and zero- aliases, and non-standard terms, as well as some misspelled words, may
shot prediction problems in ICD coding, adding the description informa- appear in medical records; thus, EMRs represent high-noise and high-
tion of the code and using graph convolution to learn according to the sparsity text. Therefore, modeling of long medical texts will inevitably
hierarchical structure of the code itself, instead of the random initializa- be affected by redundancy or errors in information, potentially leading
tion vector in the CAML model [23] . Moreover, Pengtao et al. [73] used to important information being missed.
the coding description and also introduced adversarial learning to solve To our knowledge, few approach have been proposed to specifi-
the problem of inconsistent styles between the disease description and cally solve this problem. Some specific mechanisms may be invoked
EMR text. to improve the representation of long texts. Zhou et al. [60] used a
Introducing e xternal k nowledge. To the best of our knowledge, self-distillation learning mechanism to deal with the noise problem in
the earliest work using external knowledge is that of [28] , who stored long texts. Their teacher model used the description of the target code,
the first paragraph of the text and corresponding “Signs and symptoms ” whereas the student model used the original text with noise to learn the
section in Wikipedia for each diagnosis, and employed condensed mem- ability to extract key information from a long text. Luo et al. [61] com-
ory networks to enable interaction of clinical notes with the knowledge pressed the features obtained with filters by attention; the sliding win-
base. Wikipedia was also used as an external knowledge source in the dows of filters produce redundant information, so pooling is required
work of [76] . In addition to external data sources, the hierarchical ICD to distinguish the important adjacent phrases and filter out noise. These
taxonomy itself can also be used as knowledge, as the high-level codes attempts indicate some possible directions for that could lead to break-
can provide general information for its lower-frequency low-level codes. throughs regarding this challenge, but there is a long way to go before
Tsai et al. [77] used the hierarchical category knowledge of the diagnos- satisfactory modeling of long documents is achieved.
170

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
6.4. Interpretability of coding 7.4. Medical expenses payment
One of the drawbacks of deep learning models is their poor inter- Control of unnecessary medical expenses payments is an important
pretability. However, the interpretability of the automated ICD coding application of ICD coding. DRGs based on ICD coding represent a vital
task as a decision-making assistance task is essential for its clinical appli- means of achieving effective management of medical quality and costs.
cations. In other words, it is necessary to provide corresponding support- DRGs are used to group cases with similar clinical processes and similar
ing information and decision-making assistance when predicting ICD cost consumption based on comprehensive consideration of the patient’s
codes to improve the interpretability of the model [29] . age, gender, length of stay in the hospital, clinical diagnosis, illness,
The attention mechanism is the most intuitive and widely used surgery, comorbidities, complications, etc. A system that is divided into
method to provide interpretability, as it can infer which words or frag- the same group for management.
ments the model pays more attention to when predicting. Mullenbach The implementation of DRGs can reduce the loss of medical pay-
et al. [23] used a per-label attention mechanism to learn the importance ments, ensure that the main diagnostic and treatment measures paid for
weight of words in a document; the greater the weight, the more rele- by medical insurance are focused on patients who need them most, and
vant the word to the current label. Baumel et al. [44] used sentence-level guide the use of limited medical funds to protect more people.
attention to identify sentences related to each label in a document; sen-
tences represent more coarse-grained and more complete information
7.5. Medical record management
than words.
In addition to the attention mechanism, Cao et al. [38] proposed
In EMRs, the first page of the inpatient medical record is a concise de-
the DCANM model, which uses an n-gram matching mechanism to ob-
scription of the patient’s diagnosis and treatment. Medical record man-
tain continuous word semantics and uses dilated convolution to obtain
agement is an important part of hospital management as it provides
non-continuous word semantics. The matched words are seen as in-
an abundant resource for clinical decisions, teaching, and scientific re-
terpretable information. More intuitively, Duque et al. [80] used the
search.
TF-IDF technique and medical tagger tools to extract relevant words,
The application of an ICD automatic coding system places higher re-
phrases, and medical concepts for each code from EMRs to their con-
quirements on doctors’ medical record-writing; the clarity and accuracy
structed knowledge base. The testing of the model included a mapping
of the main diagnosis on the first page of the medical record are partic-
and ranking process to complete the assignment.
ularly important. Moreover, the sharing of coding information between
clinical departments and medical record systems can help doctors to fill
7. Applications of automated ICD coding
in the information on the first pages of medical records, ensure the qual-
ity of imported information, and promote improvements in the quality
7.1. Assisting professionals in decision-making
of hospital medical record management.
The most straightforward application of automated ICD coding tasks
7.6. Hospital evaluation management
is to assist professional coders to finish the assignment decision-making
process. This frees humans from repetitive and time-consuming work
Health-related document databases can be searched by disease code
and improves the efficiency and quality of coding. Moreover, the coding
for medical record content including disease diagnosis, pathological di-
system can also be used in the auto-checking and error correction of
agnosis, surgical and operation incision classification, hospital infection
already-coded health-related documents to control the quality of ICD
diagnosis, death of patients, etc. This information can be retrieved for a
code assignment.
specific time period to evaluate the hospital’s performance during that
time. Hospital management personnel and medical quality management
7.2. Statistics and analysis on disease and death
personnel can obtain real global data on hospital management through
this path, which is beneficial to the management and standardization of
Analysis of health information statistics has an irreplaceable role in
their practice.
the development of the world’s medical and healthcare systems. The in-
Moreover, with the continuous advancement of DRGs, indicators
troduction of ICD codes in the 19th century was intended to classify and
such as the number of groups of DRGs, case-mix index, time consump-
count the causes of deaths from disease. Incidence, mortality, and sur-
tion index, cost consumption index, low-risk group mortality, and non-
vival are still key indicators for hospitals, national medical departments,
entry status are gradually being utilized to evaluate the performance
the World Health Organization, and other institutions.
of the hospital, and it makes it possible to compare the performance
Through ICD codes, statisticians can clearly analyze the occurrence
between different hospitals and different departments.
and development of diseases in a certain hospital, a certain community,
and a certain area; the types of diseases in a certain hospital, the types
of diseases, and the types of outpatient presentations; causes of death; 8. Discussion and future prospects
and reasons for injuries and poisonings. Further statistical analyses can
be implemented on outpatient data, hospitalization data, medical data, In this paper, we have introduced the importance and necessity of
management data, medicinal materials data, and economic data, and the ICD coding task, reviewed the research progress with respect to this
the results can be used to guide decision-making. task in recent years, and summarized the work on how to translate this
problem into a learnable task. The development history of automated
7.3. Medical data standardization and sharing ICD coding can be divided into three stages. The two key points when
solving this problem are the representation learning of documents to
Digital and intelligent medical treatment has become an inevitable obtain diagnosis-related information and the classification of ICD codes
trend in the development of medical management. ICD coding provides from the diagnosis. We have also summarized the main challenges to be
a standard medical information system, which is a prerequisite for digi- overcome.
tal medical care. The first problem that needs to be solved is the coding For future studies, making this task a knowledge-driven task is a
problem of medical records, so as to realize the standardization of diag- promising direction. As domain knowledge has been accumulated over
nostic and treatment information and further achieve data sharing. The many years, such as in medical KGs, using attributes of this knowledge
widespread use of ICD coding enhances communication and sharing be- to enhance our understanding of disease connotations may be beneficial
tween hospitals, regions, and countries. to coding prediction.
171

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
Another problem to be considered is practicality of the clinical ap- [20] Wang G , Li C , Wang W , et al. Proceedings of the 56th Annual Meeting of the Associ-
plications in medical systems. Potential solutions include building cod- ation for Computational Linguistics (Volume 1: Long Papers). Melbourne, Australia:
Association for Computational Linguistics; 2018. p. 2321–31 .
ing models of different scales or granularities to meet the actual coding
[21] Atutxa A, de Ilarraza AD, Gojenola K, et al. Interpretable deep learning to
needs of different scenarios; implementing conversions between differ- map diagnostic texts to icd-10 codes. Int J Med Inform 2019;129:49–59.
ent ICD versions (different languages, different hospitals), etc; and help- doi: 10.1016/j.ijmedinf.2019.05.015 .
ing the iteration of new and old versions.
[22] Xu K , Lam M , Pang J , et al. In: Kale DC, Ranganath R, Wallace BC, editors Proceed-
ings of the Machine Learning for Healthcare Conference, MLHC 2019, Ann Arbor,
Michigan, USA. PMLR; 2019. p. 197–215 .
Conflicts of interest statement [23] Mullenbach J , Wiegreffe S , Duke J , et al. Proceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies; 2018. p. 1101–11 .
The authors declare that there are no conflicts of interest. [24] Yu Y, Li M, Liu L, et al. Automatic ICD code assignment of chinese clinical
notes based on multilayer attention birnn. J Biomed Inform 2019;91:103114.
doi: 10.1016/j.jbi.2019.103114 .
Funding [25] Cao P , Chen Y , Liu K , et al. Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics. Online: Association for Computational Linguistics;
2020. p. 3105–14 .
This research was supported by Beijing Municipal Natural Science
[26] Suominen H , Kelly L , Goeuriot L , et al. In: Murtagh F, Nie JY, Soulier L, editors Ex-
Foundation (Grant No. M22012) and BUPT Excellent Ph.D. Students perimental IR Meets Multilinguality, Multimodality, and Interaction. Cham: Springer
Foundation (Grant No. CX2021122). International Publishing; 2018. p. 286–301 .
[27] Kelly L , Suominen H , Goeuriot L , et al. In: Rauber A, Müller H, Losada DE, edi-
tors Experimental IR Meets Multilinguality, Multimodality, and Interaction. Cham:
Author contributions Springer International Publishing; 2019. p. 322–39 .
[28] Prakash A , Zhao S , Hasan S , et al. Proceedings of the Thirty-First AAAI Conference
on Artificial Intelligence; 2017. p. 3274–80 .
Chenwei Yan: Conceptualization, Writing –original draft, Writing [29] Vani A , Jernite Y , Sontag D . Grounded recurrent neural networks. arXiv preprint
–review & editing, Data curation, Formal analysis. Xiangling Fu: Con- arXiv:170508557 2017 .
ceptualization, Writing – review & editing. Xien Liu: Conceptualiza- [30] Farkas R, Szarvas G. Automatic construction of rule-based icd-9-cm coding sys-
tems. BMC Bioinformatic 2008;9 Suppl 3(Suppl 3):S10. doi: 10.1186/1471-2105
tion, Writing –review & editing. Yuanqiu Zhang: Data curation, Formal -9-S3-S10 .
analysis. Ji Wu: Conceptualization, Writing –review & editing. Qiang [31] Lita LV , Yu S , Niculescu S , et al. Proceedings of the Third International Joint Con-
Li: Writing –review & editing. ference on Natural Language Processing: Volume-II; 2008. p. 877–82 .
[32] Perotte A, Pivovarov R, Natarajan K, et al. Diagnosis code assignment: models
and evaluation metrics. J Am Med Inf Assoc 2013;21(2):231–7. doi: 10.1136/amia-
References jnl-2013-002159 .
[33] Koopman B, Zuccon G, Nguyen A, et al. Automatic icd-10 classification of can-
[1] Manchikanti L . Implications of fraud and abuse in interventional pain management. cers from free-text death certificates. Int J Med Inform 2015;84(11):956–65.
Am Soc Intervent Pain Phys 2002;5(3):320–37 . doi: 10.1016/j.ijmedinf.2015.08.004 .
[2] Dee L. Consultant report-natural language processing in the health care industry. [34] Kang N, Singh B, Afzal Z, et al. Using rule-based natural language processing to im-
2007. prove disease normalization in biomedical text. J Am Med Inf Assoc 2012;20(5):876–
[3] Kaur R. Proceedings of the 57th Annual Meeting of the Association for Computa- 81. doi: 10.1136/amiajnl-2012-001173 .
tional Linguistics: Student Research Workshop. Florence, Italy: Association for Com- [35] Elyne S, Boris C, Kim L, et al. Selecting relevant features from the electronic
putational Linguistics; 2019. p. 1–9. doi: 10.18653/v1/P19-2001 . health record for clinical code prediction. J Biomed Inform 2017;74:92–103.
[4] Yang Y , Chute CG . Proceedings of the Annual Symposium on Computer Application doi: 10.1016/j.jbi.2017.09.004 .
in Medical Care. California: IEEE Computer Society; 1994. p. 157–61 . [36] Suominen H , Ginter F , Pyysalo S , et al. Machine learning to automate the assignment
[5] Larkey LS, Croft WB. Proceedings of the 19th Annual International ACM SIGIR of diagnosis codes to free-text radiology reports: a method description; 2007 .
Conference on Research and Development in Information Retrieval. SIGIR ’96. [37] Marafino BJ, Davies JM, Bardach NS, et al. N-Gram support vector machines for
New York, NY, USA: Association for Computing Machinery; 1996. p. 289–97. scalable procedure and diagnosis classification, with applications to clinical free
doi: 10.1145/243199.243276 . text data from the intensive care unit. J Am Med Inform Assoc 2014;21(5):871–5.
[6] Zhang D, He D, Zhao S, et al. BioNLP 2017. Vancouver, Canada,: Association for doi: 10.1136/amiajnl-2014-002694 .
Computational Linguistics; 2017. p. 263–71. doi: 10.18653/v1/W17-2333 . [38] Cao P , Yan C , Fu X , et al. Proceedings of the 58th Annual Meeting of the Associa-
[7] Aden. Medical record disease classification and coding defect analysis report in tion for Computational Linguistics: System Demonstrations. Online: Association for
2019, 2019, Computational Linguistics; 2020. p. 294–301 .
[8] Kumar V, Recupero DR, Riboni D, et al. Ensembling classical machine learning and [39] Ji S , Cambria E , Marttinen P . Proceedings of the 3rd Clinical Natural Language Pro-
deep learning approaches for morbidity identification from clinical notes. IEEE Ac- cessing Workshop. Online: Association for Computational Linguistics; 2020. p. 73–8 .
cess 2020;9:7107–26. doi: 10.1109/ACCESS.2020.3043221 . [40] Sadoughi N , Finley GP , Fone J , et al. Medical code prediction with multi-view con-
[9] Kumar V, Mishra BK, Mazzara M, et al. Prediction of malignant & benign volution and description-regularized label-dependent attention. arxiv 2018 .
breast cancer: a data mining approach in healthcare applications. arxiv 2019. [41] Falis M , Pajak M , Lisowska A , et al. Proceedings of the Tenth International Work-
doi: 10.48550/arXiv.1902.03825 . shop on Health Text Mining and Information Analysis (LOUHI 2019). Hong Kong:
[10] DessìD, Helaoui R, Kumar V, et al. TF-IDF Vs word embeddings for morbidity identi- Association for Computational Linguistics; 2019. p. 168–77 .
fication in clinical notes: an initial study. arxiv 2021. doi: 10.5281/zenodo.4777594 . [42] Li F, Yu H. ICD coding from clinical text using multi-filter residual con-
[11] Abhyankar S, Demner-Fushman D, Callaghan FM, et al. Combining structured and volutional neural network. Proc AAAI Conf Artif Intell 2020;34(5):8180–7.
unstructured data to identify a cohort of ICU patients who received dialysis. J Am doi: 10.1609/aaai.v34i05.6331 .
Med Inf Assoc 2014;21(5):801–7. doi: 10.1136/amiajnl-2013-001915 . [43] Mayya V, S SK, Krishnan GS, et al. Multi-channel, convolutional atten-
[12] Friedman C, Shagina L, Lussier Y, et al. Automated encoding of clinical documents tion based neural model for automated diagnostic coding of unstructured
based on natural language processing. J Am Med Inf Assoc 2004;11(5):392–402. patient discharge summaries. Future Generat Comput Syst 2021;118:374–91.
doi: 10.1197/jamia.M1552 . doi: 10.1016/j.future.2021.01.013 .
[13] Subotin M, Davis A. Proceedings of BioNLP 2014. Baltimore, Maryland: Association [44] He K , Zhang X , Ren S , et al. 2016 IEEE Conference on Computer Vision and Pattern
for Computational Linguistics; 2014. p. 59–67. doi: 10.3115/v1/W14-3409 . Recognition (CVPR); 2016. p. 770–8 .
[14] Rizzo SG , Montesi D , Fabbri A , et al. In: Ambite J-L, editor Data Integration in the [45] Shi H , Xie P , Hu Z , et al. Towards automated ICD coding using deep learning. arXiv
Life Sciences. Cham: Springer International Publishing; 2015. p. 147–61 . preprint arXiv:171104075 2017 .
[15] Kavuluru R, Rios A, Lu Y. An empirical evaluation of supervised learning ap- [46] Baumel T , Nassour-Kassis J , Cohen R , et al. Proceedings of the Workshops of the
proaches in assigning diagnosis codes to electronic medical records. Artif Intell Med Thirty-Second AAAI Conference on Artificial Intelligence; 2018. p. 409–16 .
2015;65(2):155–66. doi: 10.1016/j.artmed.2015.04.007 . [47] Guo D, Duan G, Yu Y, et al. A disease inference method based on symptom extrac-
[16] Scheurwegs E, Luyckx K, Luyten L, et al. Data integration of structured and un- tion and bidirectional long short term memory networks. Methods 2020;173:75–82.
structured sources for assigning clinical codes to patient stays. J Am Med Inf Assoc doi: 10.1016/j.ymeth.2019.07.009 .
2016;23:e11–19. doi: 10.1093/jamia/ocv115 . [48] Rios A , Kavuluru R . Proceedings of the Conference on Empirical Methods in Natural
[17] Chen Y, Lu H, Li L. Automatic icd-10 coding algorithm using an im- Language Processing, 2018; 2018. p. 3132–42 .
proved longest common subsequence based on semantic similarity. PLoS ONE [49] Du Y, Xu T, Ma J, et al. An automatic icd coding method for clin-
2017;12(3):e0173410. doi: 10.1371/journal.pone.0173410 . ical records based on deep neural network. Big Data Res 2020;6(5):0.
[18] Mario A , Raquel M , Victor F , et al. ICD-10 Coding based on semantic distance: lsi doi: 10.11959/j.issn.2096-0271.2020040 .
uned at clef ehealth 2020 task 1. Proc Conf Labs Evaluat Forum 2020;2696 . [50] Wang W , Xu H , Gan Z , et al. The Thirty-Fourth AAAI Conference on Artificial In-
[19] Ning W, Yu M, Zhang R. A hierarchical method to automatically encode chinese telligence, AAAI 2020; The Thirty-Second Innovative Applications of Artificial In-
diagnoses through semantic similarity estimation. BMC Med Inform Decis Mak telligence Conference, IAAI 2020; The Tenth AAAI Symposium on Educational Ad-
2016;16:30. doi: 10.1186/s12911-016-0269-4 . vances in Artificial Intelligence, EAAI 2020, New York, NY, USA. AAAI Press; 2020.
p. 979–88 .
172

C. Yan, X. Fu, X. Liu et al. Intelligent Medicine 2 (2022) 161–173
[51] Gao Y , Fu X , Liu X , et al. Proceedings of 2021 IEEE International Conference on [66] Johnson A , Pollard T , Shen L , et al. MIMIC-III, A freely accessible critical care
Bioinformatics and Biomedicine 2021 . database. Sci Data 2016;3:16–35 .
[52] Teng F, Yang W, Chen L, et al. Explainable prediction of medical codes with knowl- [67] Johnson A, Bulgarelli L, Pollard T, et al. MIMIC-IV (version 1.0). 2021.
edge graphs. Front Bioeng Biotechnol 2020;8:867. doi: 10.3389/fbioe.2020.00867 . doi: 10.13026/s6n6-xd98 .
[53] Chelladurai J , Santhiappan S , Ravindran B . Advances in Knowledge Discovery and [68] Pestian JP , Brew C , Matykiewicz P , et al. Proceedings of the Workshop on BioNLP
Data Mining. Cham: Springer International Publishing; 2021. p. 225–37 . 2007: Biological, Translational, and Clinical Language Processing. USA: Association
[54] Wang S , Ren P , Chen Z , et al. Few-shot electronic health record coding through graph for Computational Linguistics; 2007. p. 97–104 .
contrastive learning. arXiv 2021 . [69] Goeuriot L , Kelly L , Suominen H , et al. In: Kelly L, Goeuriot L, Mandl T, editors Ex-
[55] Devlin J, Chang MW, Lee K, et al. Proceedings of the 2019 Conference of the North perimental IR Meets Multilinguality, Multimodality, and Interaction. Cham: Springer
American Chapter of the Association for Computational Linguistics: Human Lan- International Publishing; 2017. p. 291–303 .
guage Technologies. Minneapolis, Minnesota: Association for Computational Lin- [70] Neves M, Butzke D, Dörendahl A, et al. Non-technical summaries of animal
guistics; 2019. p. 4171–86. doi: 10.18653/v1/N19-1423 . experiments indexed with icd-10 codes (version 1.0). 2019. Available from
[56] Lee J, Yoon W, Kim S, et al. Biobert: a pre-trained biomedical language repre- https://www.openagrar.de/receive/openagrar_mods_00046540 .
sentation model for biomedical text mining. Bioinformatics 2019;36(4):1234–40. [71] Goeuriot L , Suominen H , Kelly L , et al. Experimental IR Meets Multilinguality,
doi: 10.1093/bioinformatics/btz682 . Multimodality, and Interaction. Cham: Springer International Publishing; 2020.
[57] Huang K, Altosaar J, Ranganath R. ClinicalBERT: Modeling clinical notes and pre- p. 255–71 .
dicting hospital readmission. 2020. arXiv:1904.05342. [72] Subotin M, Davis AR. A method for modeling co-occurrence propensity of clin-
[58] Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining ical codes with application to ICD-10-PCS auto-coding. J Am Med Inform Assoc
for biomedical natural language processing. ACM Trans Comput Healthcare 2016;23(5):866–71. doi: 10.1093/jamia/ocv201 .
2022;3(1):1–23. doi: 10.1145/3458754 . [73] Pengtao X , Haoran S , Ming Z , et al. Proceedings of the 56th Annual Meeting of the
[59] Zhang Z , Liu J , Razavian N . Proceedings of the 3rd Clinical Natural Language Association for Computational Linguistics; 2018. p. 1066–76 .
Processing Workshop. Online: Association for Computational Linguistics; 2020. [74] Tsai SC, Huang CW, Chen YN. Proceedings of the 2021 Conference of the North
p. 24–34 . American Chapter of the Association for Computational Linguistics: Human Lan-
[60] Pascual D , Luck S , Wattenhofer R . Proceedings of the 20th Workshop on Biomedi- guage Technologies. Association for Computational Linguistics; 2021. p. 4043–52.
cal Language Processing. Online: Association for Computational Linguistics; 2021. doi: 10.18653/v1/2021.naacl-main.318 .
p. 54–63 . [75] Rubin TN , Chambers A , Smyth P , et al. Statistical topic models for multi-label doc-
[61] Luo J, Xiao C, Glass L, et al. Findings of the Association for Computational Linguis- ument classification. Mach Learn 2012;88:157–208 .
tics: ACL-IJCNLP. Association for Computational Linguistics; 2021. p. 2096–101. [76] Bai T , Vucetic S . The World Wide Web Conference; 2019. p. 72–82 .
doi: 10.18653/v1/2021.findings-acl.184 . [77] Tsai SC, Chang TY, Chen YN. Proceedings of the Tenth International Workshop on
[62] Zhou T, Cao P, Chen Y, et al. Proceedings of the 59th Annual Meeting of the Asso- Health Text Mining and Information Analysis (LOUHI 2019). Hong Kong: Associa-
ciation for Computational Linguistics and the 11th International Joint Conference tion for Computational Linguistics; 2019. p. 39–43. doi: 10.18653/v1/D19-6206 .
on Natural Language Processing. Association for Computational Linguistics; 2021. [78] Wang K , Chen X , Chen N , Chen T . Automatic emergency diagnosis with knowl-
p. 5948–57. doi: 10.18653/v1/2021.acl-long.463 . edge-based tree decoding. In: Bessiere C, editor. Proceedings of the Twenty-Ninth In-
[63] Vu T, Nguyen DQ, Nguyen A. Proceedings of the Twenty-Ninth International Joint ternational Joint Conference on Artificial Intelligence, IJCAI-20. International Joint
Conference on Artificial Intelligence 2020. doi: 10.24963/ijcai.2020/461 . Conferences on Artificial Intelligence Organization; 2020. p. 3407–14 .
[64] Xie X , Xiong Y , Yu PS , et al. Proceedings of the 28th ACM International Conference [79] Zeng M , Li M , Fei Z , Yu Y , Pan Y , Wang J . Automatic ICD-9 coding via deep
on Information and Knowledge Management. New York, NY, USA: Association for transfer learning. Neurocomputing. Deep Learning for Biological/Clinical Data
Computing Machinery; 2019. p. 649–58 . 2019;324:43–50 .
[65] Wu Y, Zeng M, Fei Z, et al. Kaicd: a knowledge attention-based deep [80] Duque A, Fabregat H, Araujo L, et al. A keyphrase-based approach for inter-
learning framework for automatic icd coding. Neurocomputing 2020. pretable icd-10 code classification of spanish medical reports. Artif Intell Med
doi: 10.1016/j.neucom.2020.05.115 . 2021;121:102177. doi: 10.1016/j.artmed.2021.102177 .
173