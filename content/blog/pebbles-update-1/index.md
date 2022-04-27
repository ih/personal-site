---
title: Pebble Graphics Update 1
date: "2022-04-27"
description: "Moving and Grooving"
---

In the [last post](../tale-of-two-modes/) I described the basic elements of Pebble Graphics and so far I've made early versions of the move and turn commands as well as loop and branch structures. I think of each piece as a computable unit. They can be run individually and stepped through for computations that are compositional like the loop.  

Below is an example of the Move command.   
<iframe width="560" height="315" src="https://www.youtube.com/embed/FgtMM2AxZTA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The green circle is a "container" that represents a distance parameter for how far the dog moves. In this case a primitive value of 10 is placed inside and the menu attached to the command is used to run it, moving the dog 10 units forward, as well as undo the execution of the command.

The Rotate command works similarly. Here hands are used to interact with the command instead of controllers, turning the dog 90 degrees when run.

<iframe width="560" height="315" src="https://www.youtube.com/embed/jlhF2QCMeEo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The loop structure is a little more complicated.

<iframe width="560" height="315" src="https://www.youtube.com/embed/V3G6JT0qhB0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Other pieces can be dragged onto the loop, move and turn commands in the video, and running the loop means continuously iterating through the elements and executing them one after another (not demonstrated here is a break command that stops the loop from executing any more items). The video also demonstrates the step buttons in the menu that allow a user to pause after each element of the loop is executed. The red sphere indicates which element is next to be run.

Finally there's the branch structure.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pxNp439u-8E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The condition at the top is computed and if the result is true the right side is then executed otherwise the left. The green sphere indicates where the branch is in its computation. 

That's what I have so far! I expect the visuals to completely change and adding variables (probably the hardest part) is yet to come, but I think it's a good start to a fun spatial interactive computing environment.