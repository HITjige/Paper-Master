# Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases

19210 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025

## Multiple Inﬂuences Maximization Under Dynamic

## Link Strength in Multi-Agent Systems: The Competitive and Cooperative Cases

Mincan Li , Zidong Wang , *Fellow,* *IEEE*, Simon J. E. Taylor , Kenli Li , *Senior* *Member,* *IEEE*, Xiangke Liao , and Xiaohui Liu product, thereby expanding its acceptance among an increasing ***Abstract *—This article addresses the issue of multiple inﬂuences** **maximization** **under** **dynamic** **link** **strength** **(MIMDLS)** **in** **multi-** number of ordinary users. The objective of IM is to ﬁnd the **agent** **systems** **(MASs).** **Initially,** **a** **novel** **model** **for** **dynamic** **link** seed users who can maximize the number of ordinary users **strength** **within** **MASs** **is** **suggested** **to** **facilitate** **the** **simulation** inﬂuenced, typically under a speciﬁc diﬀusion model. The IM **of** **multiple** **inﬂuences** **di**ﬀ**usion.** **Subsequently,** **the** **MIMDLS** problem can be considered an algorithmic problem or tackled **problem** **is** **formulated** **with** **both** **competitive** **and** **cooperative** as a discrete optimization problem, and it has been proven to **scenarios** **being** **examined.** **In** **response,** **two** **di**ﬀ**usion** **models,** be NP-hard under traditional diﬀusion models [25].

**speciﬁcally** **the** **competitive** **multiple** **inﬂuences** **independent** IM-related research has predominantly focused on three **cascade** **(Cp-MIIC)** **model** **and** **the** **cooperative** **multiple** principal strategies: approximation algorithms, heuristic meth- **inﬂuences** **linear** **threshold** **(Cr-MILT)** **model,** **are** **designed** ods, and community-based approaches. Approximation algo- **for** **MASs.** **Furthermore,** **a** **distributed** **deep** **reinforcement** rithms, which tackle IM as a combinatorial optimization [53], **learning** **(DRL)** **framework** **is** **established** **based** **on** **MASs** **by** [56], [60], [64] challenge, have oﬀered provable guarantees, **incorporating** **asynchronous** **training** **and** **updating** **processes** **for** **seed** **selection** **in** **the** **context** **of** **multiple** **inﬂuences.** **Moreover,** demonstrating a (1−1/*e*) ratio based on the greedy principle **the** **developed** **distributed** **DRL** **algorithm** **encompasses** **the** within the independent cascade (IC) model and linear thresh- **estimation** **of** ***Q*** **value** **as** **well** **as** **the** **management** **of** **constraints** old (LT) model. Consistent with these ﬁndings, an extensive **within** **Cp-MIIC** **and** **Cr-MILT** **models.** **Finally,** **comprehensive** and expanding corpus of studies has explored optimal greedy **experiments** **are** **conducted** **to:** **1)** **validate** **the** **e**ﬀ**ectiveness** **and** techniques to achieve the best possible solutions. Heuristic **e**ﬃ**ciency** **of** **the** **proposed** **models** **and** **algorithms** **in** **terms** **of** methods, on the other hand, have been favored for their scala- **multiple inﬂuence di**ﬀ**usion and 2) benchmark their performance** bility and faster execution time, as they do not necessitate the **against** **state-of-the-art** **methods.** computation of approximation bounds. Notably, metaheuristic ***Index*** ***Terms *—Deep** **reinforcement** **learning** **(DRL),** **dynamic** algorithms have simpliﬁed the complexity to *O*(*kd*(*m* + *n*)) **link** **strength,** **inﬂuence** **di**ﬀ**usion,** **multi-agent** **systems** **(MASs),** in the context of the IC model [21], where *k* is the size **multiple** **information** **maximization** **(MIM).** of seed set, *d* is the length of deep searching, *m* is the total number of edges, and *n* is the population of network.

I. INTRODUCTION The community-based approach has been introduced as a

# T

HE problem of inﬂuence maximization (IM), initially means of discovering superior solutions in comparison to introduced in [16], has emerged as a focal point in some advanced heuristic methods [10]. Moreover, integrated research on viral marketing, link prediction [18], [61], and strategies of greedy heuristic and Hop-based approaches have community detection within social networks (SNs). At its been proposed, yielding satisfactory outcomes across various core, IM seeks to harness the potential of inﬂuencers within a scenarios of information diﬀusion [48].

static network to propagate the inﬂuence of a speciﬁc topic or Over time, to better align with real-world scenarios, a diverse array of IM variants has been introduced in the litera- Received 4 February 2024; revised 1 November 2024 and 12 April 2025;

accepted 7 July 2025. Date of publication 6 August 2025; date of current ture[5],[37],[38],[40], and some of them are within dynamic version 9 October 2025. This work was supported in part by the National Key environments [1], [9]. It is important to note that most of the Research and Development Program of China under Grant 2020YFB2104000;

studies [9], [35], [40] have concentrated on the availability of in part by the National Natural Science Foundation of China under Grant links; they have largely overlooked the dynamics associated 61625202, Grant 61751204, Grant 61860206011, and Grant 62206091; in with link weights. In [40], a novel sketch-based method has part by the Natural Science Foundation of Hunan Province of China under Grant 2023JJ40166; in part by the Royal Society, U.K.; and in part by the been introduced by employing an index that adjusts sketches Alexander von Humboldt Foundation of Germany. *(Corresponding* *author:* through expanding or shrinking them, thereby facilitating the *Kenli* *Li.)* determination of*k*-coverage within a dynamic network[40]. In Mincan Li and Kenli Li are with the College of Computer Science and addition, the seed set and pseudo-seed set have been designed Electronic Engineering, Hunan University, Changsha, Hunan 410082, China, and also with the National Supercomputing Center in Changsha, Changsha, to support the T *×* oneHop approach to deal with changing Hunan 410082, China (e-mail: limc@hnu.edu.cn; lkl@hnu.edu.cn).

links among SNs [35].

Zidong Wang, Simon J. E. Taylor, and Xiaohui Liu are with the Department Recently, signiﬁcant emphasis has been placed on devel- of Computer Science, Brunel University London, Uxbridge, UB8 3PH Middle- oping behavior-aware IM strategies in SNs, recognizing that sex, U.K. (e-mail: Zidong.Wang@brunel.ac.uk; Simon.Taylor@brunel.ac.uk;

eﬀective IM solutions should account for more than just the Xiaohui.Liu@brunel.ac.uk).

Xiangke Liao is with the Collaborative Innovation Center of High Per- graph structures. In fact, it is crucial to incorporate user formance Computing, National University of Defense Technology, Changsha behaviors, activities, and their variations into the analysis.

410073, China (e-mail: xkliao@nudt.edu.cn).

Behavior-aware IM examples, such as the label-aware model Digital Object Identiﬁer 10.1109/TNNLS.2025.3588236 2162-237X © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.

<!-- Page 2 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19211 ing link weights. This oversight can largely be attributed to the fact that traditional IM and MIM investigations have tended to prioritize network topologies over the nuances of link properties and user eﬀects on links. For instance, to maximize the total inﬂuence in multiple isomorphic networks, a parallel greedy framework had been provided [54]. Con- sidering competitive commodities in real world, a model for competitive inﬂuence diﬀusion had been suggested: strategies involving known and unknown competitors have been utilized, leading to the formulation of an *n*-player diﬀusion game that aims for a Nash equilibrium [32], [63]. Nonetheless, MIM is distinctively challenged by the prevalent competitive and cooperative/coordinate relationships among inﬂuences under a dynamic circumstance, presenting a complex problem that has begun to draw signiﬁcant scholarly attention.

This article explores the problem of multiple inﬂuences maximization under dynamic link strength (MIMDLS) in MASs with applications to SNs. The key contributions are Fig. 1. Example of the general MIM problem (*k*=3).

outlined as follows.

1. A novel dynamic model is proposed for link strength,

[4], target-aware model [29], and topic-aware model [50], grounded in MAS, encompassing the generation of link underscore the pivotal inﬂuence of user behaviors on the strength, interaction protocols among agents, and the spread of inﬂuence. By extracting and examining user behavior evolving rules for dynamic link strength.

characteristics, the IM problem can be approached more

2. The concept of the MIMDLS problem is formulated to

eﬀectively. A practical method involves the creation of a multi- model and encapsulate the process of maximizing multi- agent system (MAS) model, which simulates user activities ple inﬂuences within an SN, which is further diversiﬁed and manages behavioral attributes to enhance the process of into the cases of competitive MIMDLS (Cp-MIMDLS) inﬂuence diﬀusion.

and cooperative MIMDLS (Cr-MIMDLS).

Extensive research has been conducted on IM and its

3. Based on MASs, a distributed DRL approach is devel-

various applications, then the concept of multiple inﬂuences oped for MIM with the aim to estimate and ensure maximization (MIM) [46] has been explored by a growing the optimization of the seed set, while addressing the number of researchers. An example in Fig. 1 is illustrated complexities associated with MIMDLS scenarios.

for a common MIM problem in a directed graph. Three inﬂuences are ready for diﬀusion and marked in three colors.

The structure of this article is organized in the following Three diﬀusion probabilities are set on every edge for the manner. Section II delves into the current state-of-the-art corresponding inﬂuence and marked as *p*ω1,*p*ω2, and *p*ω3.

technologies pertinent to the IM problem within the context Typically, the constraint on the number of times a node can be of SNs. Section III lays out the basic deﬁnitions related to activated is not required in a general MIM. Thus, in a general the MIM problem and its derivative cases as they apply to MIM problem, a node can be successfully activated by more SNs, along with relevant preliminary concepts. In Section IV, than one inﬂuence. With the constraints that the intersection of the communication protocols, a dynamic model for link the seed sets of diﬀerent inﬂuences should be an empty set and strength within MAS, and the diﬀusion models are thoroughly the total seed selection should be limited in *k*, the objective presented. Section V is dedicated to showcasing a variety of a general MIM is to select *k* seed nodes to maximize the of experimental outcomes, analyses of parameters, and the ﬁnal diﬀusion spread. In Fig. 1, the optimal seed set had juxtaposition of the proposed methodologies against a range of been selected (*k* =*3*): three nodes are selected as seed nodes standard benchmarks. This article is concluded in Section VI, for the ﬁrst inﬂuence (blue), the second inﬂuence (orange), where summative remarks are made and prospective avenues and the third inﬂuence (green), respectively. By diﬀusing for future investigation are identiﬁed.

three inﬂuences from three seed nodes with corresponding probabilities, the ﬁnal inﬂuence spread is 14 (ﬁve orange, ﬁve II. RELATEDWORKS green, and four blue nodes).

A multitude of past studies has established that the static Machine learning technologies have strong advantages of IM problem is an NP-hard problem under various diﬀusion eﬃciency and generalization ability to solve the MIM prob- models, including the IC model, LT model, triggering (TR) lem, especially reinforcement learning (RL) methods [33].

model, and continuous-time (CT) model. Within these inves- MIM can be solved by formulating multiple decision-making tigations, a greedy algorithm has often been employed to sequences in a discrete space using RL approaches. Further- select seed nodes up to a speciﬁed *k* parameter limit. This more, the solution of MIM can be extended to large-scale greedy approach is underpinned by the theoretical assurance and complicated networks via deep reinforcement learning (DRL) because of its characteristic oﬄine training and online provided by a nonnegative monotone submodular function.

The methodologies that utilize this greedy framework can decision-making. Thus, the RL-based method has an absolute be broadly categorized into three groups: simulation-based advantage in handling the complexity of dynamic factors in [62], proxy-based [20], and sketch-based [11] approaches (the link properties.

estimations of inﬂuence diﬀusion are computed by generating The literature on MIM had been focused on ﬁxed topolo- several sketches under a speciﬁc diﬀusion model). Despite gies [54], multiround inﬂuence diﬀusion [34], and multiple their theoretical underpinnings, these greedy methodologies networks [45], while ignoring dynamic factors such as chang-

<!-- Page 3 -->

19212 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 eﬃcient solutions [17], [33], [49]. RL approaches [27], [58] have been criticized for their extensive computational time and diminished eﬃciency in inﬂuence spread, particularly within are especially well-suited to IM issues, especially when these large network structures. The tradeoﬀbetween inﬂuence are conceptualized as combinatorial optimization problems.

diﬀusion and computational eﬃciency has been somewhat The adaptation of IM into a Markov decision process (MDP) framework [22], [24], [57], followed by the application of mitigated by the adoption of metaheuristic algorithms[52]. For RL [15], [59] to assimilate behaviors from historical network instance, a genetic algorithm has been introduced to reﬁne seed topologies, has been employed to tackle the contingency- selection through various strategies of population initialization aware IM problem [7]. In a distinct approach, an orthogonal [13]. Nevertheless, the heuristic approaches lack a theoretical paradigm has been developed to predict expected inﬂuence foundation and necessitate algorithmic design tailored to the diﬀusion using an RL algorithm, which notably obviates the speciﬁc diﬀusion model in use.

need for building the model from the ground up[28]. Further- Contrary to the static IM, dynamic inﬂuence maximization more, graph neural networks (GNNs) have been brought into (DIM) faces the signiﬁcant challenge of constantly evolving the fold to aid in addressing IM and its related applications.

user topologies. Recent studies have predominantly focused A position-aware inductive GNN model has been designed on evolutionary computation methods to address the com- to focus on the encoding of local neighborhood structures, plexities inherent in DIM. One such approach is an adaptive which leverages a set of anchor nodes to capture the positions evolutionary method that enhances the candidate solution of all nodes within the network, thereby optimizing global by pinpointing users with signiﬁcant inﬂuencing capabilities reachability[39]. A wide array of models speciﬁcally designed [31]. This method emphasizes the identiﬁcation of potential for various IM variants have been developed to suit particular inﬂuencers as a core component of the algorithm, serving situations via graph computation[8],[44]. The Celﬁe method, as an alternative to the sketch-based approach in accom- for example, has been explored to bypass the constraints inher- modating dynamic network changes. In addition, from the ent in conventional diﬀusion models by extracting inﬂuence standpoint of link structures, the notion of an “eﬀective link” representations through the analysis of diﬀusion cascade infor- has been introduced to lay the groundwork for a two-stage mation [41]. An adversarial graph embedding (GE) technique IM algorithm, which delves into the exchange of information has been implemented to address the fairness in nodes’ inﬂu- between user pairs, aiming to reﬁne the selection process enceability by sensitive attributes, which involves the creation for seed users and thereby enhance the overall quality of of a discriminator for sensitive attribute recognition while the inﬂuence network [19]. On the other hand, inspired by simultaneously training a GE auto-encoder [26]. Therefore, clustering concepts, the original network has been transformed integrating RL and GNN advantages constitute an eﬀective and into one of coarser granularity, and the DIM problem has practical solution for addressing the complexities of dynamic been approached by identifying seed users through the lens of link strength within MIM challenges.

community structure information [42]. Despite this innovative approach, the dynamic weight of links, a crucial determi- nant of inﬂuence diﬀusion, has often been overlooked. The III. PROBLEMFORMULATION current methodologies that focus on dynamic links fall short In this section, the foundational concepts related to IM are in networks characterized by dynamic link strength/weight, introduced, and the MIMDLS problem is deﬁned along with which is mainly due to the fact that the dynamics in such its associated two cases.

networks are deeply inﬂuenced by user behaviors, interactions, and preferences, which cannot be adequately addressed using standard evolutionary methods or clustering strategies.

*A.* *Preliminaries* In the realm of prior research, some eﬀorts have been made *1)* *MIM* *Problem:* Given an SN with *m* (*m* > 1) types to address the MIM problem within competitive settings with of inﬂuences, the objective is to identify *k* seed nodes that the aim to resolve real-world challenges. One such approach maximize the total number of activated nodes inﬂuenced by all is a maximization algorithm designed to circumvent com- types of inﬂuences. The SN is denoted as*G* ={*V*,*E*,*P*}, where petitive nodes by taking into account community dispersion *V* and *E* represent the sets of nodes and edges, respectively, and dynamic attributes aligned with user interests [51]. An and *P* is the set of probabilities indicating the diﬀusion IM method that focuses on the examination of homogeneous likelihood of inﬂuences between nodes. For each inﬂuence communities and the impact of inactive nodes has been type *i* (1 ≤*i* ≤*m*) and any two nodes *u* and *v* in *G*, deployed in [55] to assess weak inﬂuence among potential *p**i*(*u*,*v*) ∈*P* signiﬁes the diﬀusion probability of the *i*th nodes within SNs. Furthermore, a competitive version of the inﬂuence from node *u* to node *v*. The goal is to construct LT model has been developed for MIM, which assigns a a seed set *S* = *S*1 ∪*S*2 ∪· · · ∪*S**m* (|*S*| = *k*), ensuring that dimensional vector to each user to track the inﬂuence proba- *S*1∩*S*2∩· · · ∩*S**m* = ∅, with each *S**i* being the seed set for bility of various types[6]. In an eﬀort to consider the intricate the *i*th inﬂuence, and the total number of seeds across all sets interplay among inﬂuences (including both competitive and equal to *k*.

complementary dynamics), a deep recursive hybrid model has *2)* *Basic* *IC* *Model:* The basic IC model considers the been introduced for assessing the probabilities of inﬂuence diﬀusion of a single type of inﬂuence within a given SN *G* = between node pairs concerning products [23]. While these {*V*,*E*,*P*}. Initially, all nodes within the seed set are activated at models oﬀer potential solutions for the MIM challenge, there the ﬁrst time step. Subsequently, at the second time step, each has been scant exploration into MIM within the context of activated node has the potential to inﬂuence its neighboring dynamic link strength, which is an area that warrants urgent nodes based on the diﬀusion probability associated with the and thorough investigation.

respective link. The process permits only the nodes activated The utility and impact of learning-based methodologies in the current time step to inﬂuence others in the subsequent extend across a wide array of IM challenges, with RL frame- step. This diﬀusion continues until no additional nodes can be works [12], [47] standing out as particularly eﬀective and activated, signifying the end of the process.

<!-- Page 4 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19213 LS = {ls(*u*,*v*)} (*u*,*v* ∈*V*) encompassing all link strengths in the network, where ls(*u*,*v*) = {ls*i*(*u*,*v*)} (1 ≤*i* ≤*m*) for each link from node *u* to node *v*. These link strengths ls(*u*,*v*) are dynamically updated by the initiating user *u* throughout the diﬀusion process. The composition of the seed set must satisfy *S* = *S*1 ∪*S*2 ∪· · · ∪*S**m* (|*S*| = *k*), ensuring that *S*1 ∩*S*2 ∩· · · ∩*S**m* = ∅, with *S*1,*S*2, . . . ,*S**m* designated as the seed sets corresponding to the 1st, 2nd,..., *m*th inﬂuences, respectively.

The dynamic nature of link strength updates in the MIMDLS framework is inﬂuenced by user communications and interactions, reﬂecting the variable acceptability of inﬂu- ences by users. It should be mentioned that every user is a potential node for every inﬂuence, and when a user’s link strength is 0, it does not mean this user is a repelling user;

it only indicates that the user’s acceptability is 0 at the Fig. 2. Illustration for link strength.

current time step and the acceptability could be updated by the user in the future. Within the MIMDLS context, the key *3)* *Basic* *LT* *Model:* The basic LT model addresses the elements driving diﬀusion maximization include the dynamic spread of a single type of inﬂuence among users in an SN modiﬁcation of link weights and the interplay among the *G* = {*V*,*E*,*P*}. Each node within this network is assigned *m* types of inﬂuences. In light of the complex dynamics a threshold θ ∈(0,1). The diﬀusion process initiates from among multiple inﬂuences, two distinct cases of MIMDLS are seed nodes in an activated state, diﬀusing the inﬂuence to identiﬁed, namely, competitive and cooperative cases, which neighboring nodes according to the speciﬁed probabilities of are detailed as follows.

the corresponding edges. A node becomes activated when the cumulative inﬂuence it receives reaches or exceeds its *3)* *Competitive* *MIMDLS* *(Cp-MIMDLS):* In Cp-MIMDLS, threshold θ. The diﬀusion process concludes when there are the*m*types of inﬂuences present in the MIMDLS scenario are no further nodes that can be activated.

considered to be in competition with each other, meaning a node once activated by one inﬂuence cannot be inﬂuenced by another. The challenge in Cp-MIMDLS is to identify a seed *B.* *Problem* *Statement* set *S* = *S*1 ∪*S*2 ∪· · · ∪*S**m* (|*S*| = *k*), where each node is *1)* *Link* *Strength* *(LS):* In this article, link strength is activated by only one type of inﬂuence, aiming to maximize synonymous with diﬀusion probability.

the total number of nodes activated across all inﬂuences.

1. For a directed graph, the link strength of the edge

*4)* *Cooperative* *MIMDLS* *(Cr-MIMDLS):* In Cr-MIMDLS, from node *u* to node *v*, denoted as (*u*,*v*), is updated the *m* types of inﬂuence within MIMDLS interact coopera- by user *u* and represented as ls(*u*,*v*), where ls(*u*,*v*) = tively, allowing a node to be activated by multiple inﬂuences, {ls*i*(*u*,*v*)}(1≤*i*≤*m*). Here, ls*i*(*u*,*v*) signiﬁes the strength contrasting with Cp-MIMDLS’s competitive nature. In this of the *i*th inﬂuence as it diﬀuses from node *u* to node cooperative setting, successful activation by one inﬂuence *v*, with its value ranging within [0,1].

enhances the activation probability for others. Cr-MIMDLS

2. In the case of an undirected graph, the link (*u*,*v*) is

aims to maximize the total successful activations under two treated as two directed links: (*u*,*v*) and (*v*,*u*). Accord- conditions: 1) the seed set *S* = *S*1 ∪*S*2 ∪· · · ∪*S**m* (|*S*| = *k*) ingly, the link strengths for the link (*u*,*v*) are given by is composed of subsets for each inﬂuence and 2) each subset ls(*u*,*v*), which is updated by user *u*, and ls(*v*,*u*), which *S**i* contains an equal fraction of the total seeds, with |*S*1| = is under the control of user *v*, as illustrated in Fig. 2.

|*S*2| = · · · = |*S**m*| = *k*/*m*(*k*%*m* = 0). The reason for setting It should be noted that ls*i*(*u*,*v*) ∈[0,1], representing the the subsets to the same size is to avoid the impact of initial strength of the *i*th inﬂuence from node *u* to node *v*, can be diﬀusion advantage on the MIM result and to ensure diﬀusion understood in three distinct scenarios at the current time step.

fairness among inﬂuences.

1. If the link strength is 0, it indicates the absence of a link

*5)* *User*/*Agent:* In an MAS model for MIMDLS, each user from node *u* to node *v*.

is modeled as an agent *a**j* = {LS*j*,*F* *j*,ST*j*}, with *j* serving as

2. In the event that user *u* became an activated node in

the agent’s identiﬁer. Every agent *a**j* possesses a link strength matrix LS*j* = {ls*i*(*a**j*,*a**q*)} (1 ≤*i* ≤*m*,*q* ∈*N**a**j*), where *N**a**j* the last time step, node *v* will be successfully activated when ls*i*(*u*,*v*)=1.

represents the set of neighbor identiﬁers for *a**j*. The feature

3. Should user *u* have been activated in the last time

matrix for *m* inﬂuences is denoted by *F* *j* = { ⃗*f* *j* 1, ⃗*f* *j* 2,· · · , ⃗*f* *j* *m*}, step, the *i*th inﬂuence will be diﬀused to user *v* with where ⃗*f* *j* a probability equal to ls*i*(*u*,*v*) that is in (0,1).

1 is a row vector that captures the behavioral features of the 1st inﬂuence on *a**j*, and the dimensionality of ⃗*f* *j* Moreover, a link strength of 0 signiﬁes the dissolution of 1, the relationship between users, whereas a link strength of 1 represented by*d**f*, is determined by the database. The state set represents a scenario of “blind following action.” ST*j* ={st*j* 1,st*j* 2,· · · ,st*j* *m*} (0*j* *i* =0 or 1 and |*S T* *j*|=*m*) indicates *2)* *Multiple* *Inﬂuences* *Maximization* *Under* *Dynamic* *Link* the activation status of *a**j* with respect to each inﬂuence;

*Strength (MIMDLS):* In the context of an SN where*m*(*m*>1) st*j* *i* =1 signiﬁes that *a**j* is activated by the *i*th inﬂuence, while types of inﬂuences are present, the challenge is to identify *k* st*j* *i* =0 indicates unsuccessful activation.

seed nodes that will maximize the total number of activated The initial state set for agent *a**j* is set as ST*j* = nodes under the condition that the link strength set LS is {0, . . . ,0}(|ST *j*| = *m*) before inﬂuence diﬀusion. It should be dynamic and subject to updates by users at each time step.

The SN with LS is represented as *G*′ = {*V*,*E*,*LS*}, with *i*=1 st*j* mentioned that in a competitive scenario, the value of P*m* *i*

<!-- Page 5 -->

19214 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 Fig. 4. Example of Cp-MIIC model.

in subsequent time steps is limited to nodes that were activated in the immediate preceding step. The Cp-MIIC model diverges from the standard IC model by adhering to the following speciﬁc rules.

Fig. 3. Relationships among proposed problems and models.

1. Once activated, nodes are immune to further external

inﬂuences.

(of the state set ST *j*) is either 1 or 0, indicating that an agent

2. An activated node can propagate only the type of inﬂu-

can be activated by at most one type of inﬂuence or not ence that led to its activation.

*m*P st *j*

3. While inactive nodes may receive multiple inﬂuences

activated at all. Conversely, in a cooperative scenario, *i* from various neighbors, they can ultimately be activated *i*=1 *m*P st*j* by only a single type of inﬂuence.

*i* ≥1, allowing for can be 0, indicating no activation, or *i*=1 The distinctive aspect of the Cp-MIIC model is encapsu- the possibility of an agent being activated by one or more lated in rule 3), which mandates that if an inactive node is types of inﬂuence.

subjected to multiple inﬂuences, these inﬂuencing neighbors are arranged in a sequence based on the strength of their IV. MAINMODEL ANDAPPROACH links. The inﬂuencing process then follows this sequence, with each neighbor attempting to activate the inactive node using This section introduces diﬀusion models for multiple inﬂu- its inﬂuence until the node is either activated or all potential ences in both competitive and cooperative environments inﬂuencers have been considered. Overall, the same constraint and establishes the principal MAS model for dynamic link is reﬂected by the three rules: a node can be successfully strength, including interaction rules for MAS diﬀusion. It activated by only one inﬂuence, indicating a strict competitive concludes with the design of a distributed RL framework relationship among inﬂuences.

tailored for the MIM solution. The relationships among To demonstrate the Cp-MIIC model, consider *a*1 in Fig. 4 the proposed problems and main models are depicted in as an example of an inactive node receiving multiple types of Fig. 3. The multiple inﬂuences diﬀusions of competitive and inﬂuence at a single time step. The 1st, 2nd,., *i*th inﬂuences cooperative MIMDLS problems are simulated through MAS are simultaneously disseminated by *a*1’s activated neighbors based on diﬀusion models (Cp-MIIC and Cr-MILT), and the *a*2,*a*3, . . . ,*a**q*, all of which are adjacent to*a*1. These neighbors dynamic link strength is implemented trough agent commu- are ranked according to their link strengths with *a*1, arranged nications in MAS simultaneously. Eventually, MIMDLS is in descending order as*a*2,*a*3, . . . ,*a**q*, as depicted at the bottom solved by a distributed RL framework based on MAS through of Fig. 4. Following this sequence, the neighbors attempt to interactions with the simulation environment.

activate *a*1 with the probabilities corresponding to their link strengths. If *a*1 is successfully activated by the *i*′th inﬂuence from one of its neighbors, it will set st1 *i*′ (where st1 *i*′ ∈*S T*1) *A.* *Multiple* *Inﬂuences* *Di*ﬀ*usion* *Models* *Based* *on* *MAS* to 1, indicating activation, and will then proceed to inﬂuence Building upon the conventional diﬀusion models IC and its own inactive neighbors in the next time step. If, however, LT, two novel diﬀusion models are tailored to accommodate none of *a*1’s neighbors manage to activate it, *a*1 will remain dynamic link strength and multiple inﬂuences within both inactive state(|ST1| = 0) until the beginning of the next time competitive and cooperative settings: 1) the competitive mul- step, and it may be activated again by other agents in the next tiple inﬂuences independent cascade (Cp-MIIC) model and 2) step.

the cooperative multiple inﬂuences linear threshold (Cr-MILT) *2)* *Cr-MILT* *Model:* In the Cr-MILT model, similar to model. It should be noted that the design of the inﬂuence the conventional LT model, each agent *a**j* is assigned a diﬀusion rules determines the environment to which the model unique threshold vector θ *j*, expressed as θ *j* = {θ *j* 0, θ*j* 1,· · · , θ*j* *m*}.

can adapt. In other words, a competitive/cooperative model The values θ *j* 1, . . . , θ*j* *m* fall within the interval (0,1) and can be designed based on the LT/IC models. The reason represent the activation thresholds for each corresponding for designing the Cp-MIIC and Cr-MILT models is that it type of inﬂuence impacting *a**j*. Distinctively, θ *j* 0 ∈[1,*m*] is much easier to understand the real-world physical mean- speciﬁes the type threshold for inﬂuences on *a**j*, indicating ing (of cooperation and competition) reﬂected by these two that *a**j* can be activated by up to θ *j* 0 diﬀerent types of models.

*1)* *Cp-MIIC* *Model:* Upon the initial activation of nodes inﬂuence.

in the seed set *S* with *m* types of inﬂuence, these activated Unlike the traditional LT model, the Cr-MILT model incor- porates a cooperation vector for inﬂuences, denoted as cv = nodes proceed to activate their neighboring nodes according to {*c*1,*c*2,· · · ,*c**m*}, where each*c**i* signiﬁes the enhancement index probabilities matching the respective link strengths. Activation

<!-- Page 6 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19215

|an average of the enhan<br>After these link strength<br>proceeds analogously to S<br>is used for adjusting diff<br>step under Cr-MILT mo<br>value of LS of the nodes<br>is calculated by (2) and<br>during the diffusion proc<br>B. MAS Model for Dyna<br>Example of Cr-MILT model.<br>ith influence. For instance, if agent a is already Algorithm 1 Initializatio<br>j<br>d by the fifth influence, the link strengths for all Input: A social network G<br>ypes of influence (excluding the fifth) will receive F j.<br>ancement indexed by c at the subsequent time step. 1: for i=1 to i=m do // ith<br>5<br>c 2: for j=1 to j=|V| do|cement indices for the calculation.<br>adjustments, the diffusion process<br>ituation 1. It should be noted that (2)<br>usion probability at the current time<br>del, but does not really change the<br>. In other words, a temporary value<br>is used for threshold accumulation<br>ess under the Cr-MILT model.<br>mic Link Strength|
|---|---|
|Example of Cr-MILT model.<br> _i_th inﬂuence. For instance, if agent _aj_ is already<br>d by the ﬁfth inﬂuence, the link strengths for all<br>ypes of inﬂuence (excluding the ﬁfth) will receive<br>ancement indexed by _c_5 at the subsequent time step.<br>      _c_ <br>an average of the enhan<br>After these link strength<br>proceeds analogously to S<br>is used for adjusting diﬀ<br>step under Cr-MILT mo<br>value of LS of the nodes<br>is calculated by (2) and <br>during the diﬀusion proc<br>_B. MAS Model for Dyna_<br>**Algorithm 1** Initializatio<br>**Input:** A social network_ G_ <br>_F j_.<br>1: **for** i=1 to i=m **do** // _ith_<br>2:<br>**for** j=1 to j=|_V_| **do**|n of Link Strength for MAS Model|
|Example of Cr-MILT model.<br> _i_th inﬂuence. For instance, if agent _aj_ is already<br>d by the ﬁfth inﬂuence, the link strengths for all<br>ypes of inﬂuence (excluding the ﬁfth) will receive<br>ancement indexed by _c_5 at the subsequent time step.<br>      _c_ <br>an average of the enhan<br>After these link strength<br>proceeds analogously to S<br>is used for adjusting diﬀ<br>step under Cr-MILT mo<br>value of LS of the nodes<br>is calculated by (2) and <br>during the diﬀusion proc<br>_B. MAS Model for Dyna_<br>**Algorithm 1** Initializatio<br>**Input:** A social network_ G_ <br>_F j_.<br>1: **for** i=1 to i=m **do** // _ith_<br>2:<br>**for** j=1 to j=|_V_| **do**|= {_V_,_ E_}, every agent’s behavioral matrix<br> inﬂuence<br>// agent _a j_|

for the *i*th inﬂuence. For instance, if agent *a**j* is already activated by the ﬁfth inﬂuence, the link strengths for all other types of inﬂuence (excluding the ﬁfth) will receive an enhancement indexed by *c*5 at the subsequent time step.

(The method for calculating the enhancement index *c**i* will be elaborated later.) The diﬀusion mechanism in Cr-MILT is agent.AttentionIndex( ⃗*f* *j* 1 ,⃗*f**q* characterized by two situations: 1) the activation of an inactive 1 ) through Eq.(3);

4:

**end** **for** node and 2) the reactivation of an already activated node by 5:

**for** every *q* (*q*∈*N**a**j*) **do** 6:

additional inﬂuences. These processes are visually represented **if** *Exist*(*edge*(*a**j*,*a**q*)) **then** 7:

in Fig. 5.

*ls**i*(*a**j*,*a**q*)=agent.Normalization(α*i* *jq*);

8:

*Situation* *1:* Here, *a*1 is initially inactive, with its type **end** **if** 9:

threshold of inﬂuences denoted as θ1

0. At each time step, *a*1

**end** **for** 10:

aggregates the inﬂuence values from each type of inﬂuence **end** **for** 11:

exerted by its neighbors, comparing the aggregate with its 12: **end** **for** respective activation thresholds. For instance, the total inﬂu- ence for the *m*th type on *a*1 at a given time step might be In the MAS model for MIM, agents representing users are calculated as *ac*θ1 *m* =ls*m*(*a*3,*a*1)+ls*m*(*a**j*,*a*1). The comparison interconnected within an SN denoted as*G*. Dynamically updat- between θ1 *m* and *ac*θ1 *m* is then conducted according to ing link strength will be executed by agents before inﬂuence diﬀusion at each time step, and prior to that, the initialization θ1 *m*,*ac*θ1 *ac*θ1 *m* −θ1 /θ1       = *m*.

eva (1) *m* *m* of link strengths should be designed. The initialization of link The extent to which the accumulated threshold exceeds the strengths is a critical step to facilitate dynamic link strength original threshold is reﬂected by eva(., .). The node cannot be implementation, acknowledging that each user exhibits distinct activated successfully when eva(., .) < 0. The evaluation is behavioral features under each type of inﬂuence, which in repeated for each type of inﬂuence, resulting in a set of eval- turn inform the initial link strengths derived from feature uations eva(θ1 *i* ,*ac*θ1 *i* ) for 1 ≤*i* ≤*m*, which are then arranged vectors. The initialization depends on agent communications in descending order. The top θ1 0 values from this ordered list and observations before the process of inﬂuence diﬀusion.

The process starts by calculating the attention index α*i* are compiled into an EVA queue for *a*1. Subsequently, *a*1 is *jq* of activated by the types of inﬂuence corresponding to all positive every user *a**j* toward each neighbor *a**q* (*q*∈*N**a**j*) under the *i*th entries in the EVA queue.

inﬂuence, as shown in the following equation:

*Situation* *2:* In this situation, *a*4 is already activated by T T *jq* =[1,1, . . . ,1] ⃗*f* *j* ∥⃗*f* *q* the ﬁrst inﬂuence, meaning it can no longer be inﬂuenced α*i* .

(3) *i* *i* by the ﬁrst inﬂuence from *a*2, but it remains susceptible to Here, α*i* activation by up toθ4 0−1 other types of inﬂuence. At the current *jq* represents the attention index of user *a**j* toward its neighbor *a**q* under the *i*th inﬂuence, with [1,1, . . . ,1] being time step, *a*4 receives the second and *m*th inﬂuences through a (2*×**d**f*)-dimensional vector where all elements are 1, and three connections, each with its own link strength. Given ∥denotes the concatenation operator. The vector [1,1, . . . ,1] the cooperative nature of the inﬂuences and the enhancement index *c*1 for the ﬁrst inﬂuence, the link strengths (ls2(*a*3,*a*4), can be regarded as a weight vector for every element in T ls*m*(*a*3,*a*4), and ls*m*(*a**j*,*a*4)) are adjusted as per the following T the connected feature vector ⃗*f* *j* ∥⃗*f* *q* , and it also can be set equation:

*i* *i* with various elements according to the speciﬁc situation. The *a**j*,*a**j*′   1+ 1 attention index is calculated by (3) with both *a**j*’s feature X     *a**j*,*a**j*′   =ls*i* .

*ad* *c**p* ls*i* (2) and its neighbors. Following this, the attention indexes for *n* *a**j* are normalized to derive the initial link strength β*i* *jq* for Here, *c**p* denotes the enhancement index for the *p*th inﬂu- each neighbor *a**q* under the *i*th inﬂuence, as detailed in the ence, with *p* belonging to the set NA*a**j*′ that comprises following equation:

the identiﬁers of inﬂuences that have successfully activated    *a**j*′. The set NA*a**j*′, with cardinality *n*, represents the inﬂu- α*i* exp LeakyReLU *jq* ences that have activated *a**j*′. For example, the adjusted link β*i* *jq* = .

(4)   strength from *a*3 to *a*4 for the 2nd inﬂuence is calculated α*i* P *q*′∈*N**a j* exp LeakyReLU *jq*′ as *ad*(ls2(*a*3,*a*4)) = ls2(*a*3,*a*4)(1+*c*1). Similarly, the adjust- ment from *a*4 to *a*3 for the ﬁrst inﬂuence is determined by This equation serves as a normalization mechanism for all *ad*(ls1(*a*4,*a*3)) = ls1(*a*4,*a*3) ∗(1 + 1/2(*c*2 + *c**m*)), employing attention indexes under each type of inﬂuence, setting the

<!-- Page 7 -->

19216 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 **Algorithm** **3** Agent Interaction Algorithm initial link strength from *a**j* to *a**q* for the *i*th inﬂuence as ls*i*(*a**j*,*a**q*) = β*i* 1: **for** every *a**j* ∈*A* **do** *jq*. The comprehensive procedure is elaborated **for** t=1 to *t*=*t**max* **do** 2:

in Algorithm 1.

**if** IsNewlyActivated(*a**j*) **then** 3:

*DN* ←DiﬀusionNeighbor(*a**j*);

4:

**Algorithm** **2** Dynamic Link Strength in MAS //neighbors could be activated by *a**j* *LS* *j* ←UpdatedLinkStrength(*a**j*);

**Input:** A social network *G* = {*V*,*E*}, agent set *A* in MAS with an 5:

initialized link strength matrix of every agent.

//get the updated value of link strength 1: **for** *t*=1 to *t*=*t**max* **do** //*t**th* time step **for** every *a**j*′ ∈*DN* **do** 6:

**for** every agent *a**j* ∈*A* **do** 2:

Diﬀusing(*a**j*,*a**j*′);

7:

**if** *sum*(*S T* *j*)≥1 **then** //if it is an activated node 3:

//diﬀusing inﬂuence from *a**j* to *a**j*′ Update(*a**j*,*IN**j* ←*S T* *j*); //dynamically updating the 4:

**end** **for** 8:

corresponding link strength according to Eq.(5) **end** **if** 9:

**end** **if** 5:

**if** ActivatedNeighbor(*a**j*) **then** 10:

**end** **for** 6:

AN ←ActivatedNeighbor(*a**j*);

11:

7: **end** **for** //add to activated neighbor set of *a**j* **for** every *a**q* ∈*AN* **do** 12:

**if** (DiﬀusionCapacity(*a**q*)) **then** Link strengths, initially set based on the behavioral features 13:

⃗*f* *j* 1, ⃗*f* *j* 2, . . . , ⃗*f* *j* //if *a**q* can activated *a**j* *m* of agent *a* *j*, are dynamically updated by the *BDN* ←*a**q*;

14:

agents themselves, reﬂecting observations of their neighbors’ //add to BeDiﬀusedNodeSet of *a**j* *LS* *q* ←UpdatedLinkStrength(*a**q*);

behaviors. The process for dynamically updating link strength 15:

is outlined in Algorithm 2. At each time step, all activated **end** **if** 16:

agents update their link strengths prior to the inﬂuence dif- **end** **for** 17:

fusion interactions, as speciﬁed in line 4. The state set ST *j* BeDiﬀused(*a**j*,*BDN*,ModelType);

18:

//*a**j* accepts diﬀusion from neighbors of agent *a**j* (as deﬁned in user/agent) holds the information UpdateState(*a**j*);

19:

on whether and by which type of inﬂuence *a**j* has been **end** **if** 20:

activated, with the inﬂuence type’s identiﬁcation number *i*′ **end** **for** 21:

being retrievable from IN*j*. To be speciﬁc, if agent*a**j* has been 22: **end** **for** activated by at least one inﬂuence, the identiﬁcation numbers of inﬂuences that have successfully activated *a**j* are collected in IN*j* through ST*j*. Then, the link strengths for *a**j* under the inﬂuence type *i*′th(*i*′ ∈IN*j*) are then updated based on in DN using the relevant inﬂuences, guided by the updated link strengths storing in LS *j*. After the diﬀusion attempt, *a**j* a comparison of the corresponding features of *a**j*’s neighbors, may itself become activated by any of its activated neighbors as detailed in the following equation:

that possess the capability to diﬀuse inﬂuence, chosen from  ls*i*′   *a**j*,*a**q* the set of already activated neighbors AN, as detailed in lines 13 and 14. Specially, the function in line 13 is a boolean   8 ⃗*f* *j* *i*′, ⃗*f**q* 0, β*i*′ *p*∼*U*     ≤0.5 , *p* if*ED* < function to determine whether the agent has diﬀusion capacity.

*i*′ *jq* = (5) *p*′ ∼*U* This function works by checking activated agents: if the agent β*i*′ *p*′     , *jq*,1 otherwise :

was successfully activated by the inﬂuence(s) in the last time step but not in the step before that, it possesses diﬀusion where ED( ⃗*f* *j* *i*′, ⃗*f**q* *i*′) represents the Euclidean distance between capability; otherwise, it does not. Then, all activated neighbors the two behavioral feature vectors and is regarded a behavioral having diﬀusion capability are collected in the set BDN and similarity (under *i*′th inﬂuence) between *a**j* and *a**q*(*q* ∈*N**a**j*).

*a**j* will accept the corresponding inﬂuences diﬀused by these If the behavioral similarity is less than 0.5, the updated value neighbors under the ﬁxed diﬀusion model, line 18. The state of follows a uniform distribution in the range of [0, β*i*′ *jq*]. If the *a**j* is updated to reﬂect the results of these diﬀusion attempts, similarity is not lower than 0.5, the value follows a uniform as noted in line 19.

distribution within the interval [β*i*′ *jq*,1].

**Algorithm** **4** Multiple Inﬂuences Diﬀusion in MAS **Input** **2:** the seed set *S* = *s*1,*s*2, . . . ,*s**m*.

*C.* *Agent* *Interaction* *and* *MAS* *Inﬂuence* *Di*ﬀ*usion* 1: Initialization: algorithm 1();

In the MAS model, the process of inﬂuence diﬀusion 2: **for** i=1 to i=m **do** is facilitated by interactions between agents, as depicted in *r**i* =0;

3:

Algorithm 3. On the one hand, the diﬀusion activity for an 4: **end** **for** agent *a**j*, which was successfully activated in the preceding 5: ActivatingSeedSet(S);

time step, is detailed from lines 3 to 9 of the algorithm. On 6: **while** NewlyActivated() **do** the other hand, agent *a**j* is susceptible to activation through Dynamic link strength: algorithm 2();

7:

Agent interaction: algorithm 3();

8:

its activated neighbor agents, adhering to the predeﬁned rules **for** i=1 to i=m **do** 9:

of the chosen diﬀusion model, as indicated from lines 10 to Records(*r**i*);

10:

18. The state of *a**j* is then adjusted in accordance with the

**end** **for** 11:

outcomes of this diﬀusion process, as outlined in line 19.

12: **end** **while** In the context of the ﬁxed diﬀusion model, during the diﬀusion phase for agent*a**j*, the set DN is deﬁned to comprise the neighbors to whom*a**j* can extend its inﬂuence, as speciﬁed The setup for MAS in MIMDLS is outlined in Algorithms in line 4. *a**j* then proceeds to activate each neighboring agent 1–3, covering initialization and dynamic updates of link

<!-- Page 8 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19217 **Algorithm** **5** Distributed DRL Algorithm for MIMDLS **Input** **1:** A social network *G* = {*V*,*E*}, seed number *k*, every agent’s behavioral matrix *F* *j*.

**Input** **2:** the parameters of the global AC network: δ*g*, ω*g*, the global time step *T*, period time of every executor agent *pn**z*.

1: **for** z=1 to z=m **do** *d*δ←0, *d*ω←0; //Reset gradient of AC network z.

2:

δ′ ←δ*g*, ω′ ←ω*g*; //Synchronizing, local parameters are 3:

δ′, ω′ *t*=1; //Reset iteration number of executor *agent**z*.

4:

*s**z* =∅; *state**a**z* = *s**z* //Initialize seed set 5:

**if** !SelectionConﬂict() **then** 6:

π ←−*action**a**z* *action**a**z* *t* ;

7:

Fig. 6. Distributed DRL framework for MIMDLS.

**else** 8:

*t* ←*Compare*(*action**a**z* *action**a**z* by the horizontal arrows in Fig. 6. Each agent operates its *t* );

9:

**end** **if** own actor-critic (AC) network, which is used to evaluate *Q* 10:

*t*+1, *r**a**z* *alg*4 *state**a**z* ←*state**a**z* ←−*r**a**z* values for potential diﬀusion outcomes initiated by selected *t* ();

11:

*t*=*t*+1,*T* =*T* +1;

12:

seed nodes. The AC network also aids in aggregating gradient **if** Cp-MIIC?|*S*|==*k*:|*s**z*|==*k*/*m*||CN()==0 **then** 13:

updates for the loss function following each agent’s interaction *state**a**z* ←*state**terminal* 14:

with the environment *G*. Periodically, the agents update the **end** **if** 15:

parameters of a global AC network with their accumulated **if** *t*== *pn**z*||*state**a**z* == *state**terminal* **then** 16:

gradients and synchronize their local AC networks with this **if** *t*== *pn**z* **then** 17:

global network to align for subsequent iterations.

*Q*(*state**a**z*,*t*)=*V*(*state**a**z*(*s**z*), ω′) 18:

In the distributed DRL framework for solving MIMDLS **else** 19:

problems, the essential components, namely, *action*, *reward*, *Q*(*state**a**z*,*t*)=0 20:

*state*, and *policy*, are deﬁned as follows.

**end** **if** 21:

**else** 22:

1. *State:* The global state of the SN *G* at time step *t*,

go to line 6;

23:

denoted as State*t*, encapsulates the comprehensive set **end** **if** 24:

of current seed nodes. For each agent agent*z* dedicated **for** b=t-1 to b=1 **do** 25:

to maximizing the *z*th inﬂuence, its thread state, rep- *Q*(*state**a**z* *b* ,*b*)=*r**a**z* *b* +γ*Q*(*state**a**z* *b*+1,*b*+1);

26:

resented as state*a**z* *t* , includes the speciﬁc seed nodes *d*δ←*d*δ+▽δ′logπδ′(*state**a**z* *b* ,*action**a**z* *b* ) 27:

identiﬁed for that inﬂuence at time *t*.

*Q*(*state**a**z* *b* ,*action**a**z* +▽δ′*H*(π(*state**a**z* *b* ;δ′));

*b* ,*b*)

2. *Action:*The action taken by agent*z* involves appending a

∂(*Q*(*state**az* *b* ,*b*)−*V*(*S tate**b*,ω′))2 *d*ω←*d*ω+ ;

28:

∂ω′ new node to its designated seed set for the*z*th inﬂuence.

**end** **for** 29:

This selection is informed by a policy π, which dictates Update(δ*g*,*d*δ), Update(ω*g*,*d*ω); //Global updating 30:

the choice of node.

**if** *T* <*T**max* **then** 31:

3. *Reward:* The reward *r**a**z* resulting from an action by

go to line 2;

32:

agent*z*, i.e., the addition of a node to a seed set, is **else** 33:

Final parameters: δ*g* and ω*g*.

quantiﬁed by the number of nodes newly activated as 34:

**end** **if** 35:

a consequence of this action.

36: **end** **for**

4. *Policy:* The policy π employed in this context adheres

to a greedy approach, wherein nodes are chosen based on their potential to yield the highest*Q*value, indicative of their expected contribution to inﬂuence diﬀusion.

strengths and agent interactions. The primary algorithm for multiple inﬂuences diﬀusion in MAS is detailed in Algorithm These components collectively drive the decision-making

4. A key feature is the tracking of nonseed-activated nodes

process in the DRL algorithm, facilitating the dynamic and by each inﬂuence type, recorded in *r**i*, which also serves as a strategic selection of seed nodes to optimize inﬂuence spread reward metric in Algorithm 5.

within the network.

The distributed DRL algorithm for addressing the MIMDLS challenge is outlined in Algorithm 5. Here, the *m* executor *D.* *Distributed* *DRL* *for* *MIMDLS* agents represent individual threads interacting with the SN *G* at each time step. In line 6 of the algorithm, agents are tasked Drawing inspiration from the asynchronous advantage actor- with performing actions (speciﬁcally, adding a node to their critic (A3C) algorithm [36], a distributed DRL framework respective seed sets) while ensuring that the seed sets remain is proposed for identifying optimal seed sets in MIMDLS mutually exclusive, as indicated by*S*1∩*S*2∩· · ·∩*S**m* =∅. This problems, as depicted in Fig. 6. This framework features a mini MAS with *m* executor agents, where each agent mutual exclusivity prevents any executor agent from selecting (agent1,agent2, . . . ,agent*m*) is tasked with selecting seed nodes a node already chosen by another, necessitating interagent communication to resolve any selection conﬂicts over a node to maximize the spread of the 1st, 2nd,., mth inﬂuence, *a**j*′. During such communications, agents assess the *Q* value respectively. These *m* agents function as distributed, though of *a**j*′ under their respective inﬂuences. The agent whose interconnected, threads within the DRL system, each exploring inﬂuence yields the highest *Q* value for *a**j*′ gains the privilege seed nodes within the SN (*G*).

To ensure that the seed sets for diﬀerent inﬂuences remain to incorporate this node into its seed set. The other agents, in mutually exclusive (*S*1 ∩*S*2 ∩· · · ∩*S**m* = ∅), the agents turn, must then opt for the node with the next highest*Q* value as determined by their AC networks. Following the execution communicate with each other in each iteration, as indicated

<!-- Page 9 -->

19218 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 each user across a span of 400 days. The four behaviors are of their actions, the agents transition to their subsequent states indicated as*b*1 (commenting),*b*2 (liking),*b*3 (posting), and*b*4 and are rewarded based on the number of nodes activated as a result of the diﬀusion process within *G*.

(mentioning) in the following experiments. These interactions form the basis for constructing the SN, with user links deﬁning The iterative cycle within the distributed DRL algorithm for relationships and behavior logs contributing to feature vector the MIMDLS problem, spanning from lines 6 to 23, continues creation, each with a dimensionality of 4, reﬁned through data unless agent*z* reaches a terminal state or the designated time preprocessing.

step pn*z* is met. An agent enters a terminal state under For the purpose of these experiments, the dataset is struc- speciﬁc circumstances: if there are no more viable candidate tured to facilitate the study of diﬀusion dynamics across ﬁve nodes left in the network *G* for selection (CN() = 0), if the global seed set is fully populated with *k* seeds in the distinct types of inﬂuences. Consequently, the preprocessing phase involves segmenting each user’s behavioral data into ﬁve competitive MIMDLS scenario (under the Cp-MIIC model), or subsets, corresponding to the diﬀerent inﬂuences, as detailed if the agent’s individual seed set in the cooperative MIMDLS scenario (under the Cr-MILT model) reaches its limit of *k*/*m* in Algorithm 6. The main steps are as follows.

seeds, as outlined in lines 13–15. Upon reaching the terminal *Step* *1:* As outlined from lines 2 to 14, the behavioral data state or at the pn*z* time step, the agent ceases state transitions.

collected over 400 days is divided into ﬁve distinct sets labeled The *Q* value for the terminal state or the state at time step behavior*x*′, where *x*′ ranges from 1 to 5, each set representing pn*z* is either calculated or set to zero, detailed from lines 16 behaviors under a speciﬁc type of inﬂuence. (The behaviors to 21. The process of computing *Q* values and accumulating that occurred on the *x*th day are stored in recordsDay(*x*).) gradients for optimization occurs at each time step within the *Step* *2:* The probability of each behavior type occurring is interval [1,*t*−1], where *t* denotes the current time step, as computed by dividing the count of a speciﬁc behavior by described in lines 25–29. During this process, *r**a**z* *b* indicates the aggregate count of all behaviors, thereby normalizing the the reward of selecting a node (by agent *a**z*) in the *b*th time data. See lines 15 and 16, where |*b*1| indicates the occurrence step and its value equals to the number of nodes that are number of *b*1 behavior.

newly activated as a result of the selected node’s diﬀusion.

The concluding action of an episode involves synchronizing **Algorithm** **6** Data Preprocessing the agent-speciﬁc parameters with the global AC network (see 1: **for** every user (regard as agent) *a**j* in database **do** line 30).

**for** x=1 to x=400 **do** 2:

A key feature distinguishing the DRL approach for the **if** x%5==0 **then** 3:

MIMDLS problem is the interconnected nature of the *m* *behavior*1 ←*recordsDay*(*x*);

4:

threads, representing the*m*agents. Unlike completely indepen- **else** **if** x%5==1 **then** 5:

dent threads, these agents engage in communication to prevent *behavior*2 ←*recordsDay*(*x*);

6:

overlap in their seed node selections, ensuring that their actions **else** **if** x%5==2 **then** 7:

*behavior*3 ←*recordsDay*(*x*);

are coordinated and conﬂict-free. Moreover, the simultaneous 8:

**else** **if** x%5==3 **then** 9:

action-taking by the *m* agents on the SN *G* introduces a layer *behavior*4 ←*recordsDay*(*x*);

10:

of complexity where the reward for an action taken by one **else** 11:

agent is not isolated but can be aﬀected by the concurrent *behavior*5 ←*recordsDay*(*x*);

12:

actions of the other agents. This interdependency means that **end** **if** 13:

the outcome of any single action is a result of not just the **end** **for** 14:

individual agent’s strategy but also the collective dynamics of **for** *x*′ =1 to *x*′ =5 **do** 15:

all agents’ decisions within the network at that time.

|*b*1|/|*behavior**x*′|, *p*2 |*b*2|/|*behavior**x*′|, *p*3 = = = *p*1 16:

|*b*3|/|*behavior**x*′|, *p*4 =|*b*4|/|*behavior**x*′|;

*f* *j* *x*′ ={*p*1,*p*2,*p*3,*p*4} 17:

V. EXPERIMENTS ANDANALYSIS **end** **for** 18:

This section delves into the experimental setup, including 19: **end** **for** the environment, database, methods for comparison, and the results obtained from these tests. The speciﬁcs regarding *Step* *3:* With these probabilities computed, the feature vector the database and parameter conﬁgurations for the MIMDLS ⃗*f* *j* 1, ⃗*f* *j* 2, ⃗*f* *j* 3, ⃗*f* *j* 4, and ⃗*f* *j* 5 for each inﬂuence type are readily estab- experiments, along with the comparison methods, are detailed lished, as indicated in line 17.

in Section V-A. Following this, Section V-B evaluates the eﬀectiveness and eﬃciency of the proposed methods within the It should be mentioned that there is no data for threshold vector θ *j* (θ *j* = {θ *j* context of the Cp-MIIC model and the Cr-MILT model, across 0, θ*j* 1,· · · , θ*j* *m*}) of each agent *a* *j* under Cr- a variety of experimental scenarios. The comparative perfor- MILT model, but thresholds can be initialized through original link strength generated via Algorithm 1. The value of θ *j* for mance analysis, juxtaposing the proposed methods against both baseline and cutting-edge techniques, is provided in Sec- *a**j* is calculated by the following equations:

tionV-C. The implementation of the proposed algorithms was !

   *a**j*,*a**q* carried out using C++ and Python programming languages ls*i* θ *j*    *q*∈*N**a**j* *i*=1,2, . . . ,*m*.

.

*i* =Avg and executed on a CPU-based computing infrastructure.

P*m*    *a**j*,*a**q* ρ=1 lsρ (6) *A.* *Database,* *Parameters* *Settings,* *and* *Compared* *Methods* It can be seen that the value of θ *j* *i* is the average link strength *1)* *Database:* The experimental database utilizes real-world of the *i*th inﬂuence across all out-degrees of *a**j* network data from SinaWeibo, gathered using Python crawlers, |*N**a j*| 0 0 1 1 encompassing interactions among 5000 users. This dataset not θ *j* X only includes user connections but also tracks four behavioral    ≥0.5 0 =**count** *a**q*,*a**j* ls*i* A? 1 : 0 (7) @ @ A activities (commenting, liking, posting, and mentioning) for *q*=1

<!-- Page 10 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19219 where*i*=1,2, . . . ,*m*,*q*∈*N**a**j*, and**count**() is a count function, when the summation of link strength of the*i*th inﬂuence across all in-degrees of *a**j* exceeds 0.5, the function will count 1;

otherwise it will count 0.

*2)* *Parameter* *Settings:* For the experiments involving dif- fusion of multiple inﬂuences, in line with the database setup described earlier, ﬁve distinct types of inﬂuence are consid- ered, with each user’s behavior represented by a 4-D feature vector. The parameters for the MIMDLS experiments are conﬁgured as follows.

1. The number of inﬂuence types *m* is set to 5, aligning

Fig. 7. Percentages of activated nodes in diﬀerent settings under Cp-MIIC.

with the number of distinct inﬂuences being studied.

2. The seed set *S* is structured as a collection of subsets

{*S*1,*S*2,*S*3,*S*4,*S*5}, each corresponding to one of the ﬁve types of inﬂuence.

3. The feature matrix for each agent *F* *j* comprises ﬁve

4, and ⃗*f* *j* 3, ⃗*f* *j* 2, ⃗*f* *j* 1, ⃗*f* *j* feature vectors ⃗*f* *j* 5, reﬂecting the user’s behaviors under each inﬂuence type; the ﬁve feature vectors are generated through data preprocessing in Algorithm 6.

4. The dimensionality *d**f*

of the ﬁve feature vectors ( ⃗*f* *j* 1, ⃗*f* *j* 2, . . . , ⃗*f* *j*

5. of one user is set to 4, consistent with

the number of behavioral metrics considered.

Fig. 8. Percentages of activated nodes in diﬀerent settings under Cr-MILT.

In the distributed DRL framework tailored for the MIMDLS scenario, there are ﬁve executor agents within the miniature type in each iteration, ensuring an even distribution of seed MAS, each responsible for optimizing the selection of seed nodes across all inﬂuence types.

nodes for one of the inﬂuence types. The discount factor γ *Community-Based**[43]**:*This method assesses the inﬂuence utilized in the DRL algorithm is set to 0.95, indicating the potential of community-centric nodes to identify key inﬂu- degree to which future rewards are considered in the current encers. Seed candidate sets are determined heuristically at the value estimation. The dimension of the state for each agent community level, with the ﬁnal seed set compiled from these agent*z* in the DRL model varies based on the diﬀusion model preliminary selections.

being applied: it is *k* for the Cp-MIIC model, indicating the *Play-Strategy* *Method* *[32]**:* Incorporating game-theoretic total number of seed nodes across all inﬂuences, and *k*/5 for principles, this strategy treats each inﬂuence type as a player the Cr-MILT model, reﬂecting an equitable distribution of seed in a competitive setting. Seed node selection is informed by nodes among the ﬁve types of inﬂuence.

anticipating and countering the strategies of other inﬂuences, Algorithm 4, embodying the core logic of inﬂuence diﬀu- aiming for an optimal response in a competitive inﬂuence sion within this framework, has undergone training using the diﬀusion scenario.

SinaWeibo dataset, with experiments conducted across various *CoreQ* *[2]**:* Relying on *K*-core hierarchies to divide the scales of the network, including subsets of 500, 1000, 2000, network topologies and guide the identiﬁcation of seed nodes, 3000, and 5000 users. This progressive approach enables an this approach optimizes the selection of seed nodes through a evaluation of the algorithm’s performance and scalability in Q-learning algorithm.

relation to network size.

*MIM-Reasoner* *[14]**:* Combining RL with a probabilistic *3)* *Comparison* *Methods:* In the comparison experiments graphical model, the MIM-Reasoner analyzes the complex for the MIMDLS problem, several approaches are evaluated diﬀusion within a layered network and optimizes the seed alongside the proposed method.

set for each layer through RL methods. To address the MIM *GE-Based* *Method* *[30]**:* This approach utilizes GE to problem, SN can be divided into multiple layers according to represent each node’s features, incorporating neighborhood the number of inﬂuences.

topology into a new vector. The potential for a node to *MAS-Based* *DRL* *Method:* A MAS-based DRL method has spread inﬂuence is estimated using the GE-based method, with a single AC network compared with the proposed methods. In subsequent aggregation of feature information and estimation this method, only one agent is set for DRL, which exchanges to reﬁne the embedding vector and guide seed selection for information with the MAS environment (simulated by the IM. Since the GE-based method is not directly applicable proposed models). Thus, this agent set in DRL will select to MIM, MIM is decomposed into separate IM problems, seed nodes for all inﬂuence.

with the ultimate seed selection drawn from the aggregated *B.* *Experiments* *on* *Various* *Settings* *for* *Proposed* *Models* solutions of these IM instances.

*Max-k* *Coverage* *[3]**:* Employing a concept akin to reverse The evaluation of the proposed models and algorithms is inﬂuence sampling, this method begins by generating several conducted with user populations of 500, 1000, 2000, 3000, reverse reachable sets. It then addresses the maximum *k* and 5000 from the SinaWeibo dataset, employing snowball coverage problem across these sets by dynamically updating sampling for network extraction. Seed set sizes are varied the incremental value to identify the most inﬂuential nodes.

among 10, 25, 50, 75, and 100 to test the models Cp-MIIC and *Basic* *Greedy* *[25]**:* Serving as a foundational approach, Cr-MILT. Results are averaged over 100 runs and presented in this method selects nodes based on their maximum inﬂuence Fig. 7 for Cp-MIIC and Fig. 8 for Cr-MILT, showcasing the spread. For MIM challenges, it chooses one node per inﬂuence diﬀusion eﬀectiveness under diﬀerent settings.

<!-- Page 11 -->

19220 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 TABLE I PROPERTIES OFFIVENETWORKS Fig. 9. Performance of ﬁve inﬂuences under Cp-MIIC.

Fig. 11. Performance on various networks under the two models.

Fig. 10. Performance of ﬁve inﬂuences under Cr-MILT.

In Fig. 7, the average diﬀusion percentages under Cp-MIIC Fig. 12. Comparison of various methods under Cp-MIIC model.

with various settings are provided. Notably, the poorest perfor- mance occurs when*k*=10, which is attributed to the relatively low percentage of the seed set. For an agent population of 5000, achieving a diﬀusion percentage of 43% corresponds to a seed set size of *k*% = 0.02% (*k* = 100). Conversely, in a network of 500 agents, the diﬀusion percentage exceeds 85% when*k* >50, and nearly all nodes are activated when*k* =150 due to *k*% = 0.3%. The proposed methods yield satisfactory Fig. 13. Comparison of various methods under Cr-MILT model.

performance when *k*% > 1.5%.

The results under the Cr-MILT model are displayed in ﬁve (*m* = 5) inﬂuences is included in the Facebook database Fig. 8. A distinct diﬀerence from Fig. 7 is the higher dif- and does not need to be calculated.

fusion percentages. This phenomenon is caused by the fact The performance of ﬁve networks under two diﬀusion that activated nodes are counted based on the number of models is provided in Fig. 11. When *k* > 150 under the successful activations. Notably, when the population size is four networks (except for Epinions), it can be seen that the 500 and *k* ≥100, the average diﬀusion percentage surpasses percentage remains at a high level, ranging from 75% to 89% 1, implying that numerous nodes have been activated multiple under the Cp-MIIC model and from 85% to 95% under the times.

Cr-MILT model. Especially, when *k* reaches 200 under the The performance of the ﬁve types of inﬂuence under the Epinions network, the percentage of activated nodes almost two models is depicted in Figs. 9 and 10, with the population reaches 70% under the Cp-MIIC model. When *k*≥150 under setting at 5000. In Fig.9, it is easy to ﬁnd that the percentages the Cr-MILT model, the percentage ranges from 70% to 82%.

of diﬀusion for the ﬁve inﬂuences ﬂuctuate signiﬁcantly com- The reason may be that the large scale of the network requires pared with Fig. 10. A competitive result is reﬂected by Fig. 9 more seed nodes to obtain a percentage of more than 80%.

because only one node can be activated once in the spread of Cp-MIIC.

*C.* *Comparison* *With* *Advanced* *Methods* The other four networks (Facebook, Campus Forums, Flickr, and Epinions) are taken into consideration to verify the Several advanced methods are compared with the proposed eﬀectiveness of the proposed method under ﬁve inﬂuences dif- method, the distributed DRL algorithm, using a population fusion. The related properties of these networks are displayed size of 5000 users and diﬀusing ﬁve inﬂuences under the two in TableI. All links among the nodes of the four networks are diﬀusion models. After conducting each approach 100 times, collected in the database, along with their behaviors described the average performance is presented in Figs. 12 and 13.

in Table I, except for the Facebook network. Especially, Comparing the results of the Cp-MIIC model, the worst the Facebook database has 4039 nodes, each with its links performance is observed from the play-strategy and greedy and a 40-D feature. The feature is divided into ﬁve 8-D approaches in Fig.12. In the Cp-MIIC model, the competitive vectors (indicating ⃗*f* *j* 1, ⃗*f* *j* 2, ⃗*f* *j* 3, ⃗*f* *j* 4, and ⃗*f* *j* relationship among inﬂuences can be managed through the

5. for multiple inﬂuence

play-strategy method, but the dynamics of MIMDSL cannot be diﬀusion (*m* = 5). Thus, the feature matrix of each node for

<!-- Page 12 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19221 resolved. With the increasingly growing seed set, a decreasing TABLE II diﬀusion range is displayed by the play-strategy method when ITERATION LENGTH OF INFLUENCE DIFFUSION OF SOLUTIONS FROM *k* ≥150. Furthermore, anticipating and countering in the VARIOUSMETHODS play simulation become complicated due to dynamic LS, thus the precision of the calculation for play proﬁts has been aﬀected. In the greedy approach, optimal seed node computed in the current step may perform worse in the next step after inﬂuences diﬀusion, especially after signiﬁcant changes have occurred among the LS. A waving line is presented by the greedy approach because the topology structure is concentrated on the greedy idea, but the competition among users and inﬂuences are ignored. However, the calculation precision of the topology structure is compromised by dynamic link strength, which leads to an unstable result.

Although the GE-based algorithm focuses on topology structure, the information embedding and aggregation help to improve diﬀusion to around 76%. The result of the GE-based algorithm reﬂects the point that the eﬀect of dynamic LS can be partially tackled through the method of information embed- ding and aggregation, although the eﬀect is not satisﬁed. Better results are provided by the MIM-Reasoner than by the GE- Cr-MILT, a middle position is occupied by the max-*k* idea based algorithm because layers corresponding to the respective with an orange line. But the dynamics lead to a decreasing inﬂuence are tackled simultaneously by the MIM-Reasoner.

tendency when *k* > 150. With the increasing number of *k*, Worse performance is obtained by the CoreQ approach, but it more subsets of the seed set will be calculated along with their is still higher than that of the max-*k* and community-based corresponding incremental values via the max-*k* algorithm.

methods. The reason is the optimization of seed selection However, every calculation suﬀers from dynamic LS, which through the Q-learning algorithm in CoreQ. The MIMDLS is decreases the positive eﬀect of incremental value. MAS-based divided into several subgraphs by max-*k*and community-based DRL excesses GE-based and community-based approaches approaches, and the diﬀusion percentage is between 70% and when*k*<200, because the dynamic LS can be tackled well by 80% when *k* ≥150. It can be seen that within asynchronous a single AC network. Besides, this situation has not remained training in the AC networks, dynamic LS can be handled until*k*≥200, because large iterations cannot ensure parameter skillfully by the distributed DRL framework. The *Q* value convergence to an optimal value, leading to a worse quality of optimal seed nodes is estimated precisely by the trained of seed set. In contrast, the proposed distributed DRL method global AC network in every step, adapting the corresponding maintains a stable performance, achieving an activation range inﬂuence diﬀusion simulated by agent interactions. The best of 85%–95% of nodes.

results are shown by the distributed DRL framework with an Next, the performance of various technologies is analyzed increased tendency of a green line, and the maximum diﬀusion based on the diﬀusion speed. The diﬀusion speed is reﬂected is about 86%. Both competitive relationship and dynamics by the iteration length, which refers to the number of diﬀusion are taken into consideration, and a stable diﬀusion solution is iterations from the activation of seed nodes until no further generated for whatever *k*. Compared with the distributed DRL nodes can be activated. It should be noted that the iteration framework, lower diﬀusion percentages are obtained by MAS- length is recorded by simulating the diﬀusion of multiple based DRL. A single AC network has the disadvantage that the inﬂuences in MAS, based on the seed set generated by each training model sometimes is hard to ensure convergence. This approach, respectively. The longer the iteration length, the is the reason why MAS-based DRL’s performance is worse slower the diﬀusion speed. Each method is run to generate than distributed DRL.

the optimal seed set, and then multiple inﬂuence diﬀusion The corresponding results for Cr-MILT are depicted in is simulated according to the seed set to obtain the iteration Fig. 13. All performance outcomes are relatively stable due length. These steps are repeated 100 times, and the results are to the cooperative relationships among inﬂuences and the shown in TableII. For instance, based on the 100 results of the condition that each node can be activated multiple times. The GE-based algorithm under the Cp-MIIC model, after 100-time results of the play-strategy algorithm are due to the excessive simulation of diﬀusion, the average iteration length is 52, the calculations focused on agent beneﬁts in the play simulation.

maximum is 84, and 23 is the minimum value.

The improvement of cooperative MIM is gently hindered by Under the Cp-MIIC model, max-*k*, CoreQ, and basic greedy competitive play strategies. However, higher percentages are techniques have shorter average iteration lengths (33, 33, and generated by the CoreQ method, which beneﬁts from Q-

34. than others, meaning that diﬀusion is highly likely to

learning optimization. The greedy method is more eﬀective complete at a fast speed according to their optimal seed sets.

under the Cr-MILT model than under the Cp-MIIC model, However, max-*k* and greedy methods also have the lower because the fairness of cooperative inﬂuences is ensured by the diﬀusion percentages than others in Figs. 12 and 13. The greedy idea. Although relatively higher results are provided GE-based and Play-strategy methods have longer average by the MIM-Reasoner approach, it still cannot exceed 80% iteration lengths (52 and 44), while the remaining approaches because the layered calculation cannot handle the dynamic have medium average iteration lengths (distributed DRL ranks LS. The GE-based method is more eﬃcient than the MIM- sixth). The slowest speed is caused by the community-based Reasoner, which reﬂects that the representation of the graph approach with an iteration length of 91, while the fastest is structure is more appropriate than the layered algorithm under with max-*k* at 16. Besides, the shortest performance of the the Cr-MILT model. Beneﬁting from the loose constraints of

<!-- Page 13 -->

19222 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 TABLE III networks for inﬂuences are executed in parallel in distributed RUNNING TIME (MINUTES) FROM VARIOUS METHODS UNDER FIVE DRL, whereas only serial computation is performed on a INFLUENCES(*k*=150) single deep neural network in MAS-based DRL. By the way, the training time is signiﬁcantly reduced by distributed DRL, to be not more than one-third of that of MAS-based DRL.

VI. CONCLUSION This article has presented a distributed DRL framework within an MAS to address the MIMDLS problem in SNs by utilizing Cp-MIIC and Cr-MILT models. The approach has leveraged user behavior vectors to dynamically model link strengths and devise an MAS-based diﬀusion strategy for multiple inﬂuences, incorporating speciﬁc interaction rules.

proposed method is 21. The GE-based, MIM-Reasoner, and The distributed nature of MAS has inspired the development MAS-based DRL methods are the closest to the proposed of a DRL model for MIMDLS, which has been shown to method in terms of diﬀusion percentage under the Cp-MIIC be both eﬀective and eﬃcient. The model facilitates asyn- model, and they exhibit similar performance in terms of chronous updates by agents, who communicate to meet the maximum and minimum iteration length. Under the Cr-MILT requirements of diﬀerent diﬀusion models. When compared model, CoreQ’s performance is similar to MIM-Reasoner’s.

with ﬁve advanced methods, our proposed algorithm has Besides, MAS-based DRL and play-strategy methods have consistently outperformed others across a range of settings.

the closest performance to the proposed method in terms of Experimental outcomes have aﬃrmed the proposed models diﬀusion percentage, but they have longer average iteration and algorithms’ capability to eﬀectively tackle the MIMDLS lengths compared with distributed DRL (which ranks second).

problem. While the proposed approach has shown promise The longest performance is 124, generated by the GE-based for MIM problems, its reliance on predeﬁned node feature method, and the shortest is 20, generated by the basic greedy vectors is a noted limitation. Future research could explore method. The shortest iteration length of distributed DRL is 43.

MIM challenges where node information is partially known The time complexity analysis of the distributed DRL algo- or evolving, expanding the applicability of these models and rithm (see Algorithm 4) can be estimated under two diﬀusion algorithms.

models. In the Cr-MIMDLS model, the worst-case scenario is that the *Q* value of each nonseed node will be recomputed by REFERENCES the executor agent during every selection, and no node reaches [1] C. C. Aggarwal, S. Lin, and P. S. Yu, “On inﬂuential node discovery in its threshold of inﬂuence number during the selection. Thus, dynamic social networks,” in *Proc.* *SIAM* *Int.* *Conf.* *Data* *Mining*, Apr.

the upper bound of the complexity is *O*(*k*(2*n*−*k*+*m*)/2*m*).

2012, pp. 636–647.

Besides, in the Cr-MIMDLS model, the worst circumstances [2] W. Ahmad and B. Wang, “A learning-based inﬂuence maximization framework for complex networks via K-core hierarchies and reinforce- are that one executor agent selects *k* seed nodes and the *Q* ment learning,”*Expert Syst. Appl.*, vol. 259, Jan. 2025, Art. no. 125393.

value of each nonseed node is updated in the agent’s every [3] C. Borgs, M. Brautbar, J. Chayes, and B. Lucier, “Maximizing social iteration. The resulting upper bound of the complexity is:

inﬂuence in nearly optimal time,” in*Proc. 24th Annu. ACM-SIAM Symp.* *O*(*n*+*k*(1−*k*)/2). In order to further evaluate the proposed *Discrete* *Algorithms*, Jan. 2014, pp. 946–957.

methods, the running time is provided in this section.

A. Cali`o and A. Tagarelli, “Attribute based diversiﬁcation of seeds for [4] targeted inﬂuence maximization,” *Inf.* *Sci.*, vol. 546, pp. 1273–1305, Running time of various methods under diﬀerent population Feb. 2021.

sizes are displayed in Table III. Here, the size of the seed set [5] M. Charikar, Y. Naamad, and A. Wirth, “On approximating target is 150, and each result is the average time after running the set selection,” in *Approximation,* *Randomization,* *and* *Combinatorial* corresponding algorithm 30 times. The max-*k* approach has *Optimization.* *Algorithms* *and* *Techniques* *(APPROX*/*RANDOM* *2016)*.

Dagstuhl, Germany: Schloss Dagstuhl—Leibniz-Zentrum f”ur Infor- a low running time in all situations, but a worse seed set is matik, 2016.

generated by this approach compared with other methods. The [6] B. Chen, Y. Shen, M. Ji, J. Liu, Y. Yu, and Y. Zhang, “Competitive reason max-*k* uses less time is that subsets of the current seed inﬂuence maximization on online social networks under cost constraint,” set constitute the search space, which accelerates the search *KSII* *Trans.* *Internet* *Inf.* *Syst.*, vol. 15, no. 4, pp. 1263–1274, 2021.

speed, and its speed has little help for inﬂuence diﬀusion.

[7] H. Chen, W. Qiu, H.-C. Ou, B. An, and M. Tambe, “Contingency-aware inﬂuence maximization: A reinforcement learning approach,” in *Proc.* The CoreQ method has a slightly longer running time than *37th* *Conf.* *Uncertainty* *Artif.* *Intell.*, 2021, pp. 1535–1545.

max-*k* because of the optimization steps. Similar performance [8] T. Chen, J. Guo, and W. Wu, “Graph representation learning for is displayed by the MIM-Reasoner. The reason is that it can popularity prediction problem: A survey,” *Discrete* *Math.,* *Algorithms* tackle network layers in parallel and conduct batch inference.

*Appl.*, vol. 14, no. 7, Oct. 2022, Art. no. 2230003.

The structural feature is extracted by the GE-based algorithm, [9] X. Chen, G. Song, X. He, and K. Xie, “On inﬂuential nodes tracking in dynamic social networks,” in *Proc.* *SIAM* *Int.* *Conf.* *Data* *Mining*, Jun.

which achieves a low running time when the population is no 2015, pp. 613–621.

more than 3000, but the running time increases signiﬁcantly [10] Y. Chen, W. Zhu, W. Peng, W. Lee, and S. Lee, “CIM: Community- when the SN has 5000 users. The worst performance is based inﬂuence maximization in social networks,” *ACM* *Trans.* *Intell.* observed when applying the Basic Greedy method, where *Syst.* *Technol.*, vol. 5, no. 2, pp. 1–31, 2014.

the time reaches 103 due to heavy calculations in each [11] E. Cohen, D. Delling, T. Pajor, and R. F. Werneck, “Sketch-based inﬂuence maximization and computation: Scaling up with guarantees,” iteration of the complex inﬂuence diﬀusion. Running times in *Proc.* *23rd* *ACM* *Int.* *Conf.* *Conf.* *Inf.* *Knowl.* *Manage.*, Nov. 2014, of community-based and play-strategy approaches are slightly pp. 629–638.

longer. The complicated community division due to dynamic [12] Q. Cui, K. Liu, Z. Ji, and W. Song, “Sampling-data-based distributed LS and greedy calculation takes time in the Community-based optimisation of second-order multi-agent systems with PI strategy,” *Int.* *J.* *Syst.* *Sci.*, vol. 54, no. 6, pp. 1299–1312, 2023.

algorithm. The convergence process becomes lengthy due to [13] A. R. D. Silva, R. F. Rodrigues, V. D. F. Vieira, and C. R. Xavier, dynamic LS when the play-strategy method simulates play “Inﬂuence maximization in network by genetic algorithm on linear proﬁts among inﬂuences. The reason that distributed DRL threshold model,” in*Proc. 18th Int. Conf. Comput. Sci. Appl.*, Jan. 2018, takes less time than MAS-based DRL is that several AC pp. 96–109.

<!-- Page 14 -->

LI et al.: MIM UNDER DYNAMIC LINK STRENGTH IN MASs: THE COMPETITIVE AND COOPERATIVE CASES 19223 [14] N. H. K. Do, T. Chowdhury, C. Ling, L. Zhao, and M. T. Thai, “MIM- [37] R. Narayanam and Y. Narahari, “A Shapley value-based approach to discover inﬂuential nodes in social networks,” *IEEE* *Trans.* *Autom.* *Sci.* reasoner: Learning with theoretical guarantees for multiplex inﬂuence maximization,” in *Proc.* *27th* *Int.* *Conf.* *Artif.* *Intell.* *Statist.*, 2024, *Eng.*, vol. 8, no. 1, pp. 130–147, Jan. 2011.

pp. 2296–2304.

[38] H. T. Nguyen, P. Ghosh, M. L. Mayo, and T. N. Dinh, “Social inﬂuence spectrum at scale: Near-optimal solutions for multiple budgets at once,” [15] S.-S. Dong, Y.-G. Li, and L. An, “Optimal strictly stealthy attacks *ACM* *Trans.* *Inf.* *Syst.*, vol. 36, no. 2, pp. 1–26, Apr. 2018.

in cyber-physical systems with multiple channels under the energy [39] S. Nishad, S. Agarwal, A. Bhattacharya, and S. Ranu, “GraphReach:

constraint,” *Int.* *J. Syst.* *Sci.*, vol. 54, no. 13, pp. 2608–2625, Oct. 2023.

Position-aware graph neural network using reachability estimations,” in [16] P. Domingos and M. Richardson, “Mining the network value of *Proc.* *30th* *Int.* *Joint* *Conf.* *Artif.* *Intell.*, Aug. 2021, pp. 1527–1533.

customers,” in *Proc.* *7th* *ACM* *SIGKDD* *Int.* *Conf.* *Knowl.* *Discovery* [40] N. Ohsaka, T. Akiba, Y. Yoshida, and K.-I. Kawarabayashi, “Dynamic *Data* *Mining*, Aug. 2001, pp. 57–66.

inﬂuence analysis in evolving networks,” *Proc.* *VLDB* *Endowment*, [17] J. Fang, W. Liu, L. Chen, S. Lauria, A. Miron, and X. Liu, “A survey vol. 9, no. 12, pp. 1077–1088, Aug. 2016.

of algorithms, applications and trends for particle swarm optimization,” [41] G. Panagopoulos, F. D. Malliaros, and M. Vazirgianis, “Inﬂuence max- *Int.* *J.* *Netw.* *Dyn.* *Intell.*, vol. 2, no. 1, pp. 24–50, Mar. 2023.

imization using inﬂuence and susceptibility embeddings,” in *Proc.* *Int.* [18] W. Fang, B. Shen, A. Pan, L. Zou, and B. Song, “A cooperative stochas- *AAAI* *Conf.* *Web* *Social* *Media*, vol. 14, May 2020, pp. 511–521.

tic conﬁguration network based on diﬀerential evolutionary sparrow [42] X. Qin, C. Zhong, and H. X. Lin, “Community-based inﬂuence search algorithm for prediction,” *Syst.* *Sci.* *Control* *Eng.*, vol. 12, no. 1, maximization using network embedding in dynamic heterogeneous Dec. 2024, Art. no. 2314481.

social networks,” *ACM* *Trans.* *Knowl.* *Discovery* *Data*, vol. 17, no. 8, [19] B. Fu, J. Zhang, H. Bai, Y. Yang, and Y. He, “An inﬂuence maximiza- pp. 1–21, Sep. 2023.

tion algorithm for dynamic social networks based on eﬀective links,” [43] J. Shang, S. Zhou, X. Li, L. Liu, and H. Wu, “CoFIM: A community- *Entropy*, vol. 24, no. 7, p. 904, Jun. 2022.

based framework for inﬂuence maximization on large-scale networks,” [20] S. Galhotra, A. Arora, and S. Roy, “Holistic inﬂuence maximization:

*Knowl.-Based* *Syst.*, vol. 117, pp. 88–100, Feb. 2017.

Combining scalability and eﬃciency with opinion-aware models,” in [44] Y. Shang et al., “Popularity prediction of online contents via cascade *Proc.* *Int.* *Conf.* *Manag.* *Data*, 2016, pp. 743–758.

graph and temporal information,” *Axioms*, vol. 10, no. 3, p. 159, Jul.

[21] S. Galhotra, A. Arora, S. Virinchi, and S. Roy, “ASIM: A scalable

2021. 

algorithm for inﬂuence maximization under the independent cascade [45] S. S. Singh, K. Singh, A. Kumar, and B. Biswas, “MIM2: Multiple model,” in*Proc. 24th Int. Conf. World Wide Web*, May 2015, pp. 35–36.

inﬂuence maximization across multiple social networks,” *Phys.* *A,* *Stat.* [22] P. Gao, C. Jia, and A. Zhou, “Encryption–decryption-based state estima- *Mech.* *Appl.*, vol. 526, Jul. 2019, Art. no. 120902.

tion for nonlinear complex networks subject to coupled perturbation,” [46] H. Sun, X. Gao, G. Chen, J. Gu, and Y. Wang, “Multiple inﬂuence *Syst.* *Sci.* *Control* *Eng.*, vol. 12, no. 1, Dec. 2024, Art. no. 2357796.

maximization in social networks,” in *Proc.* *10th* *Int.* *Conf.* *Ubiquitous* [23] H. Huang, Z. Meng, and H. Shen, “Competitive and complementary *Inf.* *Manage.* *Commun.*, Jan. 2016, pp. 1–8.

inﬂuence maximization in social network: A follower’s perspective,” [47] Y. Sun, M. Chen, K. Peng, L. Wu, and C. Liu, “Finite-time adaptive *Knowl.-Based* *Syst.*, vol. 213, Feb. 2021, Art. no. 106600.

optimal control of uncertain strict-feedback nonlinear systems based on [24] F. Jin, L. Ma, C. Zhao, and Q. Liu, “State estimation in networked fuzzy observer and reinforcement learning,” *Int.* *J.* *Syst.* *Sci.*, vol. 55, control systems with a real-time transport protocol,” *Syst.* *Sci.* *Control* no. 8, pp. 1553–1570, Jun. 2024.

*Eng.*, vol. 12, no. 1, Dec. 2024, Art. no. 2347885.

[48] J. Tang, X. Tang, and J. Yuan, “An eﬃcient and eﬀective hop-based [25] D. Kempe, J. Kleinberg, and ´E. Tardos, “Maximizing the spread of approach for inﬂuence maximization in social networks,” *Social* *Netw.* inﬂuence through a social network,” *Theory* *Comput.*, vol. 11, no. 4, *Anal.* *Mining*, vol. 8, no. 1, pp. 1–19, Dec. 2018.

pp. 105–147, 2015.

[49] L. Tian, “Multi-dimensional adaptive learning rate gradient descent [26] M. Khajehnejad, A. Asgharian Rezaei, M. Babaei, J. Hoﬀmann, optimization algorithm for network training in magneto-optical defect M. Jalili, and A. Weller, “Adversarial graph embeddings for fair inﬂu- detection,” *Int.* *J.* *Netw.* *Dyn.* *Intell.*, vol. 3, Sep. 2024, Art. no. 100016.

ence maximization over social networks,” in *Proc.* *29th* *Int.* *Joint* *Conf.* [50] S. Tian, S. Mo, L. Wang, and Z. Peng, “Deep reinforcement learning- *Artif.* *Intell.*, Jul. 2020, pp. 4306–4312.

based approach to tackle topic-aware inﬂuence maximization,”*Data Sci.* [27] C. Li, Y. Liu, M. Gao, and L. Sheng, “Fault-tolerant formation consensus *Eng.*, vol. 5, no. 1, pp. 1–11, Mar. 2020.

control for time-varying multi-agent systems with stochastic communi- [51] J. Tong, L. Shi, L. Liu, J. Panneerselvam, and Z. Han, “A novel cation protocol,”*Int.* *J. Netw.* *Dyn.* *Intell.*, vol. 3, no. 1, Mar. 2024, Art.

inﬂuence maximization algorithm for a competitive environment based no. 100004.

on social media data analytics,” *Big* *Data* *Mining* *Anal.*, vol. 5, no. 2, [28] H. Li, M. Xu, S. S. Bhowmick, J. S. Rayhan, C. Sun, and J. Cui, pp. 130–139, Jun. 2022.

“PIANO: Inﬂuence maximization meets deep reinforcement learning,” [52] C.-W. Tsai, Y.-C. Yang, and M.-C. Chiang, “A genetic NewGreedy *IEEE Trans. Computat. Social Syst.*, vol. 10, no. 3, pp. 1288–1300, Mar.

algorithm for inﬂuence maximization in social network,” in *Proc.* *IEEE*

2023. 

*Int.* *Conf.* *Syst.,* *Man,* *Cybern.*, Oct. 2015, pp. 2549–2554.

[29] L. Li, Y. Liu, Q. Zhou, W. Yang, and J. Yuan, “Targeted inﬂuence maxi- [53] Y. Wang, W. Liu, C. Wang, F. Fadzil, S. Lauria, and X. Liu, “A novel mization under a multifactor-based information propagation model,”*Inf.* multi-objective optimization approach with ﬂexible operation planning *Sci.*, vol. 519, pp. 124–140, May 2020.

strategy for truck scheduling,” *Int.* *J.* *Netw.* *Dyn.* *Intell.*, vol. 2, Jun.

[30] M. Li et al., “Inﬂuence maximization in multiagent systems by a graph 2023, Art. no. 100002.

embedding method: Dealing with probabilistically unstable links,”*IEEE* [54] G. Wu, X. Gao, G. Yan, and G. Chen, “Parallel greedy algorithm to *Trans.* *Cybern.*, vol. 53, no. 9, pp. 6004–6016, Sep. 2023.

multiple inﬂuence maximization in social network,”*ACM Trans. Knowl.* [31] W. Li, Y. Hu, C. Jiang, S. Wu, Q. Bai, and E. Lai, “ABEM: An *Discovery* *Data*, vol. 15, no. 3, pp. 1–21, Jun. 2021.

adaptive agent-based evolutionary approach for inﬂuence maximization [55] X. Xie, J. Li, Y. Sheng, W. Wang, and W. Yang, “Competitive inﬂuence in dynamic social networks,” *Appl.* *Soft* *Comput.*, vol. 136, Mar. 2023, maximization considering inactive nodes and community homophily,” Art. no. 110062.

*Knowl.-Based* *Syst.*, vol. 233, Dec. 2021, Art. no. 107497.

[32] S.-C. Lin, S.-D. Lin, and M.-S. Chen, “A learning-based framework [56] Y. Xue et al., “Many-objective simulation optimization for camp location to handle multi-round multi-party inﬂuence maximization on social problems in humanitarian logistics,” *Int.* *J.* *Netw.* *Dyn.* *Intell.*, vol. 3, networks,” in *Proc.* *21st* *ACM* *SIGKDD* *Int.* *Conf.* *Knowl.* *Discovery* Sep. 2024, Art. no. 100017.

*Data* *Mining*, Aug. 2015, pp. 695–704.

[57] X. Yi, Z. Wang, S. Liu, and Q. Tang, “Acceleration model considering [33] Y. Li, H. Gao, Y. Gao, J. Guo, and W. Wu, “A survey on inﬂuence multi-stress coupling eﬀect and reliability modeling method based maximization: From an ML-based combinatorial optimization,” *ACM* on nonlinear Wiener process,” *Quality* *Rel.* *Eng.* *Int.*, vol. 40, no. 6, *Trans.* *Knowl.* *Discovery* *Data*, vol. 17, no. 9, pp. 1–50, Nov. 2023.

pp. 3055–3078, May 2024.

[34] I. Lozano-Osorio, J. S´anchez-Oro, and A. Duarte, “A variable neigh- [58] X.-J. Yi, C.-H. Xu, X.-T. Fang, and S.-L. Liu, “A new reliability analysis borhood search approach for the adaptive multi round inﬂuence method for software-intensive systems with degradation accumulation maximization problem,” *Social* *Netw.* *Anal.* *Mining*, vol. 14, no. 1, eﬀect based on goal oriented methodology,” *Quality* *Rel.* *Eng.* *Int.*, pp. 1–18, Aug. 2024.

vol. 40, no. 1, pp. 236–260, May 2023.

[35] Y. Meng, Y. Yi, F. Xiong, and C. Pei, “T*×*one hop approach for dynamic [59] X. Yi and T. Xu, “Distributed event-triggered estimation for dynamic inﬂuence maximization problem,” *Phys.* *A,* *Stat.* *Mech.* *Appl.*, vol. 515, average consensus:

A perturbation-injected privacy-preservation pp. 575–586, Feb. 2019.

scheme,” *Inf.* *Fusion*, vol. 108, Apr. 2024, Art. no. 102396.

[36] V.

Mnih et al., “Asynchronous methods for deep reinforce- [60] X. Yi, H. Yu, and T. Xu, “Solving multi-objective weapon-target *Proc.* *Int.* *Conf.* *Mach.* *Learn.*, ment learning,” in 2016, assignment considering reliability by improved MOEA/D-AM2M,”*Neu-* pp. 1928–1937.

*rocomputing*, vol. 563, Jan. 2024, Art. no. 126906.

<!-- Page 15 -->

19224 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 36, NO. 10, OCTOBER 2025 [61] R. Zhang, H. Liu, Y. Liu, and H. Tan, “Dynamic event-triggered state group that has contributed to the area for around 30 years. His research estimation for discrete-time delayed switched neural networks with has focused on digital infrastructures and modeling and simulation and has constrained bit rate,” *Syst.* *Sci.* *Control* *Eng.*, vol. 12, no. 1, Dec. 2024, impacted over three million students and 300 universities in Africa, over Art. no. 2334304.

30 small and medium enterprises (SMEs), and several large enterprises, [62] C. Zhou, P. Zhang, W. Zang, and L. Guo, “On the upper bounds of spread including Ford Motor Company, London, U.K., and Sellaﬁeld, Cumbria, for greedy algorithms in social network inﬂuence maximization,” *IEEE* U.K. He continues to be interested in advances in modeling and simulation, *Trans.* *Knowl.* *Data* *Eng.*, vol. 27, no. 10, pp. 2770–2783, Oct. 2015.

international development and open science, as well as helping younger [63] Y. Zhu, D. Li, and Z. Zhang, “Minimum cost seed set for competitive colleagues to rapidly start their careers.

social inﬂuence,” in *Proc.* *35th* *Annu.* *IEEE* *Int.* *Conf.* *Comput.* *Com-* Dr. Taylor co-founded U.K. Operational Research Society’s *Journal* *of* *mun.*, Apr. 2016, pp. 1–9.

*Simulation* and U.K. Simulation Workshop series and continues to be an [64] Z. Zhao, L. Xia, L. Jiang, Q. Ge, and F. Yu, “Distributed bandit online editor at the journal. He is the Former Chair and a member of the ACM optimisation for energy management in smart grids,” *Int.* *J.* *Syst.* *Sci.*, SIGSIM’s Steering Committee, an Advisory Board Member of the NSCU vol. 54, no. 16, pp. 2957–2974, Dec. 2023.

Simulation Archive, and the Executive Chair of the International Simulation Exploration Experience. He has been in various program committee positions in the IEEE/ACM Winter Simulation Conference for over 20 years and is the General Chair of WSC 2025.

**Mincan** **Li** received the Ph.D. degree in computer

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_0.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_0.png)

science and technology from the College of Com- puter Science and Electronic Engineering, Hunan University, Changsha, China, in 2021.

She is currently a Post-Doctoral Researcher with **Kenli** **Li** (Senior Member, IEEE) received the Ph.D.

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_1.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_1.png)

the College of Computer Science and Electronic degree in computer science from the Huazhong Uni- Engineering, Hunan University. Her research inter- versity of Science and Technology, Wuhan, China, ests include multi-agent systems, data engineering, in 2003.

many-objective optimization, and machine learning.

He was a Visiting Scholar with the University of Illinois at Urbana–Champaign, Urbana-Champaign, USA, from 2004 to 2005. He is currently a Cheung Kong Professor of computer science and technology with Hunan University (HNU), Changsha, China, where he is also the Vice-President, and the Dean **Zidong** **Wang** (Fellow, IEEE) received the B.Sc.

of the College of Information Science and Electronic

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_2.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_2.png)

degree in mathematics from Suzhou University, Engineering. He is also the Director of the National Supercomputing Center Suzhou, China, in 1986, and the M.Sc. degree in in Changsha, Changsha. He has published more than 350 research papers applied mathematics and the Ph.D. degree in electri- in international conferences and journals, such as IEEE TRANSACTIONS cal engineering from Nanjing University of Science ON COMPUTERS (TC), IEEE TRANSACTIONS ON PARALLEL AND DIS- and Technology, Nanjing, China, in 1990 and 1994, TRIBUTEDSYSTEMS(TPDS), International Symposium on High Performance respectively.

Computer Architecture (HPCA), International Symposium on Computer He is currently a Professor of dynamical systems Architecture (ISCA), International Conference for High Performance Comput- and computing with the Department of Computer ing, Networking, Storage, and Analysis (SC), ACM International Conference Science, Brunel University of London, Uxbridge, on Multimedia (MM), AAAI Conference on Artiﬁcial Intelligence (AAAI), U.K. From 1990 to 2002, he held teaching and Design Automation Conference (DAC), and International Conference on Data research appointments in universities in China, Germany, and U.K. He has Engineering (ICDE). His major research interests include parallel and dis- published a number of articles in international journals. His research interests tributed processing, high-performance computing, and big data management.

include dynamical systems, signal processing, bioinformatics, and control Dr. Li is a fellow of CCF. He is currently serving or has served as theory and applications.

an Associate Editor for IEEE TC, IEEE TRANSACTIONS ON INDUSTRIAL Prof. Wang is a member of the Academia Europaea and European Academy INFORMATICS (TII), and IEEE TRANSACTIONS ON SUSTAINABLE COM- of Sciences and Arts, an Academician of the International Academy for PUTING (TSUSC).

Systems and Cybernetic Sciences, a fellow of the Royal Statistical Society, and a member of the program committee for many international conferences.

He holds the Alexander von Humboldt Research Fellowship of Germany, the JSPS Research Fellowship of Japan, and the William Mong Visiting **Xiangke** **Liao** received the B.S. degree in com-

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_3.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_3.png)

Research Fellowship of Hong Kong. He serves (or has served) as the puter science and technology from the Department Editor-in-Chief for*International Journal of Systems Science*,*Neurocomputing*, of Computer Science and Technology, Tsinghua and *Systems* *Science* *and* *Control* *Engineering*, and an Associate Editor for University, Beijing, China, in 1985, and the M.S.

12 international journals, including IEEE TRANSACTIONS ON AUTOMATIC degree in computer science and technology from the CONTROL, IEEE TRANSACTIONS ON CONTROL SYSTEMS TECHNOLOGY, National University of Defense Technology, Chang- IEEE TRANSACTIONS ONNEURALNETWORKS ANDLEARNINGSYSTEMS, sha, China, in 1988.

IEEE TRANSACTIONS ONSIGNALPROCESSING, and IEEE TRANSACTIONS He is currently a Full Professor and the Dean of ON SYSTEMS, MAN, AND CYBERNETICS—PART C: APPLICATIONS AND the College of Computer Science, National Univer- REVIEWS.

sity of Defense Technology. His research interests include parallel and distributed computing, high- performance computer systems, operating systems, cloud computing, and networked embedded systems.

**Simon** **J.** **E.** **Taylor** received the B.Sc. degree

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_4.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_4.png)

(Hons.) in industrial studies and the M.Sc. degree in **Xiaohui** **Liu** received the B.Eng. degree in com-

![Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_5.png](Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_files/Multiple_Influences_Maximization_Under_Dynamic_Link_Strength_in_Multi-Agent_Systems_The_Competitive_and_Cooperative_Cases_p15_5.png)

computer studies from Sheﬃeld Hallam University, puting from Hohai University, Nanjing, China, in Sheﬃeld, U.K., in 1986 and 1988, respectively, 1982, and the Ph.D. degree in computer science from and the Ph.D. degree in distributed simulation from Heriot-Watt University, Edinburgh, U.K., in 1988.

Leeds Metropolitan University, Leeds, U.K., in He is currently Professor of computing with

1993. 

Brunel University of London, Uxbridge, U.K., He is currently Professor of computing and the where he conducts research in artiﬁcial intelli- Vice-Dean (Research) of the College of Engineering, gence, data science, and optimization, with appli- Design and Physical Sciences, Brunel University of cations in diverse areas, including biomedicine and London, Uxbridge, U.K. He is also the Co-Director engineering.

of the Modelling and Simulation Group, Brunel University of London,a
