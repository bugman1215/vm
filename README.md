# 古文翻译并排版
## Set up running environment
~~~
conda create env1
conda activate env1
~~~
## Download project
### Download project
~~~ 
git clone https://github.com/bugman1215/vm/
pip install -r requirements.txt
~~~
### Download txt file from Gutenberg
~~~
wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=txt"
~~~
![Gutenberg](https://user-images.githubusercontent.com/71434769/199416478-040bccfd-9c6d-405d-a078-01ec70ec8ea4.png)
## create new txt file to save the translation, change file path and run (optional)
~~~ 
cd vm
cat >bookname_translate.txt
vi main.py
~~~
### change file path as below suggests
~~~
Line6 : f = open("bookname.txt", encoding="utf-8")
Line46: Note = open('bookname_translate.txt', mode='w')
~~~

## Preprocessing
### output translation
~~~
python main.py
~~~
### merge 2 files
change file_path as above suggests
~~~
python merge.py
~~~
## Compile in latex
### install required latex compilers
for smaller books overleaf is fine, for larger books see [texlive](https://www.tug.org/texlive/)
## Compile and Save as bookname.pdf in Texlive
See Archive/main.tex for more information



