# 古文翻译并排版
## Set up running environment 

## Download txt file from Gutenberg


## change path in main.py and run (also need to create a new text file to save the translation)

Line6 : f = open("west.txt", encoding="utf-8")
Line46: Note = open('westtranslate.txt', mode='w')
~~~ 
pip install
~~~
## Preprocess txt files
At first I tried to compile the original book and the translation individually using \switchcolumn only once, then as the text volume grew I came to realize that latex compilation had limits on too many floats. Therefore I did python script to merge 2 txt files together, chapter to chapter to avoid the problem.  
## Compile in latex until you get the book. (For longer books it might take a while)


