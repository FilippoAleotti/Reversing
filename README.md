# Reversing the cycle

Code for "Reversing the cycle: self-supervised deep stereo through enhanced monocular distillation", ECCV 2020

# Authors
[Filippo Aleotti](https://filippoaleotti.github.io/website), [Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), [Li Zhang](), [Matteo Poggi](https://mattpoggi.github.io/), [Stefano Mattoccia](http://vision.deis.unibo.it/~smatt/Site/Home.html)

# Abstract
In many fields, self-supervised learning solutions are rapidly evolving and filling the gap with supervised approaches. This fact occurs for depth estimation based on either monocular or stereo, with the latter often providing a valid source of self-supervision for the former.
In contrast, to soften typical stereo artefacts, we propose a novel self-supervised paradigm reversing the link between the two. Purposely, in order to train deep stereo networks, we distill knowledge through a monocular completion network. This architecture exploits single-image clues and few sparse points, sourced by traditional stereo algorithms, to estimate dense yet accurate disparity maps by means of a consensus mechanism over multiple estimations. We thoroughly evaluate with popular stereo datasets the impact of different supervisory signals showing how stereo networks trained with our paradigm outperform existing self-supervised frameworks. Finally, our proposal achieves notable generalization capabilities dealing with domain shift issues.

![](assets/qualitative.jpg)

From top to bottom, the left image, the right image and the predicted disparity map. No ground-truths or LiDaR data have been used to train the network.

# Framework

![](assets/framework.jpg)

Code will be released soon
