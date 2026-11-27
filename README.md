# Attribute-Aware Crowd Generation

**[Project page](https://jungyu0413.github.io/ACG/)** · [Video](https://jungyu0413.github.io/ACG/#video)

[JunGyu Lee](https://jungyu0413.github.io/)\*, Chaemin Hwang\*, [Jisu Shin](https://jsshin.com/), [Seunghyun Shin](https://www.seunghyunshin.com/), [Hae-Gon Jeon](https://scholar.google.com/citations?user=Ei00xroAAAAJ&hl=ko)
(\* equal contribution) · Yonsei University, GIST

![Attribute-aware crowd generation: people with different attributes move and interact in different ways](docs/figs/teaser.png)

Crowds differ from place to place, and behavior differs from person to person. For example, children often walk with their parents around schools, while phone users tend to pay less attention to their surroundings. Capturing this diversity is essential for realistic crowd simulation. However, since datasets that provide both scene maps and pedestrian attribute annotations remain limited, existing crowd generation models do not explicitly learn attribute-dependent patterns. In this paper, we introduce an attribute-aware crowd generation pipeline. To do this, we first construct a crowd trajectory dataset with world-coordinate trajectories, scene semantic maps, and annotations for pedestrian attributes selected based on social studies. Building on this dataset, we propose a framework that models how attribute distributions vary across scenes and how these attributes form agents' motion patterns. Here, we incorporate two attention mechanisms: a scene cross-attention for predicting attributes from agent origins and scene maps, and an attribute-aware social attention for capturing the relative influence of each neighbor based on both agents' attributes and their relative positions. Extensive experiments demonstrate that our model outperforms baselines on public datasets as well as our attribute-annotated dataset. As an application, robot navigation policies trained on our generated crowds achieve lower collision rates while maintaining comparable success rates. To the best of our knowledge, our pipeline is the first attempt to enable attribute-aware crowd generation.

![Overview of ACG](docs/figs/overview.png)

## Status

The paper is under review. The preprint, the code and the CAT dataset will be released here.

## What is in this repository

- `docs/`: the project page, served at https://jungyu0413.github.io/ACG/, with the clips of the demonstration video.
