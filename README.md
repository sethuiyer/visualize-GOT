# visualize-GOT
This code attempts to visualise the Game of thrones dataset in kaggle, Particularly [T-SNE Visualization](https://indico.io/blog/visualizing-with-t-sne/)

### Question 1: 
To which class is the NaN of the battles[last\_row][attacker\_outcome_column] is similar to? Win or Lose?

Executing `python visualise_battle.py`, we have this following graph 

![attacker](attacker.png)

So, Mostly it is similar to the **win** outcome.

### Question 2:
How allegiances, nobility and the appearence in the book affect the gender of charecter deaths.
![figure](figure_1.png)

We see some outliers in the death of male charecters.

### Question 3:
Which feature is more powerful in the prediction? Popularity or the fact that charecter is actually alive or not?

![figure](figure_2.png)

Here, we see clusters are forming among the classes which have the common theme of popularity. Hence, we can say popularity is much more a powerful feature in the prediction.