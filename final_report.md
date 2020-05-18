The report should describe the “lessons learned”:

#### Group Organization

The various tasks varied in their ability to be split into subtasks. For the MLP- and SVM-classifiers, we decided to split into two groups,
in which each group member developed their own solution. During this, we cross-validated our results and exchanged methods. For the two 
following exercises, where such division of labor was not possible, we increased the amount of zoom-meetings to 2-3 meetings per week to compensate for a less efficient workflow. The work was cut into smaller tasks in which one or more team-members were responsible for. These tasks were usually done in 1-3 days, and the progress was discussed in the next meeting. If there were any issues during code-development, we communicated by text and helped each other out.

We didn't utilise the possibility to work on the same files using different branches as we do not feel as comfortable working
this way. Because of this, we created files with variations of the code and incorporated them into the main file. We also always screened
each others code before fitting it into the main file.

#### General evaluation of group organization

The large amount of meetings helped to verify where everyone was in their individual progress. Reviewing each other's code so often helped having a clear view for every team member on the way the code was constructed. We also realised that the organization improved for tasks 3 and 4. This is probably due to the decreased possibility to split the work and the need for more communication within the group. for task 2, we lost some time by coding each task twice, however, this helped each team-member to learn and gain experience on their own.

#### Task 2a

As mentioned before, the tasks a-c were done twice by two members each. We used the sklearn-library for both solutions. We applied the 
learning algorithm with a linear and an RBF-kernel, and tuned for the parameter C. Parameter tuning was accomplished using a grid search, for which we also utilised a function from sklearn.
#### Task 2b
As mentioned before, the tasks a-c were done twice by two members each. We thus used the tensorflow library and the sklearn library
respectively. The solution using sklearn was handed in. We optimized for the parameters hidden layer size, the amount of hidden layers, 
the learning rate, and the maximal amount of iterations.

Some parameters are already handled by the libraries. For example, there are ways to adjust the learning rate during learning, which reduces the significance of the initial learning rate. Also, in sklearn, the learning stops when the loss does not improve significantly anymore.

#### Task 2c

#### Task 3

• For each task:
• What is special about your solution

• What was your approach

• What worked, what did not work

We had some issues with handling missing values. If there are no black pixels in a window, how would we signify the non-existent upper and lower contour? Using 'None' at first needed a lot of complicated build-arounds, and thus we decided to use 0 instead, as it should not have any mathematical implications.

• General thoughts about the group exercise

#### Task 4
• For each task:
• What is special about your solution

• What was your approach

• What worked, what did not work

• General thoughts about the group exercise
