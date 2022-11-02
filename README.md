# 古文翻译并排版
## Set up running environment 
I used windows but ubuntu is fine大学生哪有不疯的
## Download project
### Download txt file from Gutenberg
~~~
wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=txt"
~~~
![Gutenberg](https://user-images.githubusercontent.com/71434769/199416478-040bccfd-9c6d-405d-a078-01ec70ec8ea4.png)
### Download project
~~~ 
git clone https://github.com/bugman1215/vm/
~~~

## create new txt file to save the translation, change file path and run (optional)
~~~ 
cd vm
cat >bookname_translate.txt
vi main.py
~~~
### change file path as below suggests

Line6 : f = open("bookname.txt", encoding="utf-8")
Line46: Note = open('bookname_translate.txt', mode='w')


## Preprocessing
### txt files
### install fonts and packages

At first I tried to compile the original book and the translation individually using \switchcolumn only once, then as the text volume grew I came to realize that latex compilation had limits on too many floats. Therefore I did python script to merge 2 txt files together, chapter to chapter to avoid the problem.

## Compile in latex
## Save as .pdf



