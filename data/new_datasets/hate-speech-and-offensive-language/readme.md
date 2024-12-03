https://github.com/t-davidson/hate-speech-and-offensive-language

# Data

The data are stored as a CSV and as a pickled pandas dataframe (Python 2.7). Each data file contains 5 columns:

`count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

`hate_speech` = number of CF users who judged the tweet to be hate speech.

`offensive_language` = number of CF users who judged the tweet to be offensive.

`neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

`class` = class label for majority of CF users.
  0 - hate speech
  1 - offensive  language
  2 - neither







2019 NEWS

We have a new paper on racial bias in this dataset and others, you can read it here

WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.

You can find our labeled data in the data directory. We have included them as a pickle file (Python 2.7) and as a CSV. You will also find a notebook in the src directory containing Python 2.7 code to replicate our analyses in the paper and a lexicon in the lexicons directory that we generated to try to more accurately classify hate speech. The classifier directory contains a script, instructions, and the necessary files to run our classifier on new data, a test case is provided.

Please cite our paper in any published work that uses any of these resources.

@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }

Contact We would also appreciate it if you could fill out this short form if you are interested in using our data so we can keep track of how these data are used and get in contact with researchers working on similar problems.

If you have any questions please contact thomas dot davidson at rutgers  dot edu.