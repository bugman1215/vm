# 古文翻译并排版
## 1 Set up running environment
### 1.1 Create virtual environment
~~~
conda create env1
conda activate env1
~~~
### 1.2 Clone project repository
~~~ 
git clone https://github.com/bugman1215/vm/
pip install -r requirements.txt
~~~
### 1.3 Download txt file from Gutenberg
~~~
wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=txt"
~~~
![Gutenberg](https://user-images.githubusercontent.com/71434769/199416478-040bccfd-9c6d-405d-a078-01ec70ec8ea4.png)
## 2 Book translation 
### 2.1 change file_path
~~~ 
cd vm
cat >bookname_translate.txt
vi main.py
~~~
change file path as below suggests
~~~
Line6 : f = open("bookname.txt", encoding="utf-8")
Line46: Note = open('bookname_translate.txt', mode='w')
~~~
### 2.2 output translation
~~~
python main.py
~~~
### 2.3 merge 2 files
change file_path as above suggests
~~~
cat >bookname_merge.py
python merge.py
~~~
## 3 Compile in latex
### 3.1 install required latex compilers
for smaller books overleaf is fine, for larger books see [texlive](https://www.tug.org/texlive/)
## 3.2 Compile and Save as bookname.pdf in Texlive
upload required files and compile main.tex for the result



