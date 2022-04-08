# Harbinger

The repository contains 6 files that are required for the complete solution.


-------------------------------------------------------------First file: Requirements.txt -------------------------------------

This file contains all the packages that need to be installed for the program to run along with the versions.

-------------------------------------------------------------Second File: Tweet_terms.csv -------------------------------------

This file contains the terms that are used by the code to extract tweets. This file can be edited to change the terms/ keywords for which the tweets are extracted.

-------------------------------------------------------------Third File: Tweet_clean_terms.csv --------------------------------

This file contains terms to remove junk tweets. A lot of tweets extracted using relevant keywords are political in nature or talk about topics that aren't related to digital payments. The most common and reccurring topics have been added to this file to remove unwanted tweets.

--------------------------------------------------------------Fourth File: Entity_cleaner.csv ---------------------------------

It contains a list of entities that are to be removed from the data. These entities are the additional entities that are removed in addition to the entities detected in the program.

--------------------------------------------------------------Fifth File: Yugank_Harbinger_PS4_solution.ipynb ------------------

This is the solution file with the complete code. It was created using a Jupyter notebook in an Anaconda distribution. This file is recommended to be used for running the program.

Anaconda distribution is recommended to run the program. Open a Jupyter notebook on Anaconda, paste this file along with the above 3 csv files in the same folder which will also be your directory. Import the packages mentioned in the Requirements.txt file. Run all the statements. 

Note: For changing the working directory use the following code:
      import os
      os.chdir(path of folder)

Note: For installing the packages use the following code:
      pip install package_name
      
Once the complete code is run, it takes around 10 mins for the complete processing. At the bottom you will see a link. Press the same to open the dashboard

---------------------------------------------------------------Sixth File: Yugank_Harbinger_PS4_solution.py ---------------------

This is the same as the above file but in different extentsion. (.py instead of .ipynb). It is recommended to use the .ipynb file
