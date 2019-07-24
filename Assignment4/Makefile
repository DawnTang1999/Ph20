.PHONY : pdfs

pdfs : explicit.pdf implicit.pdf symplectic.pdf 

symplectic.pdf : euler_methods.py 
	python euler_methods.py --Method s 

explicit.pdf : euler_methods.py
	python euler_methods.py --Method e

implicit.pdf : euler_methods.py
	python euler_methods.py --Method i


.PHONY : clean
clean :
	rm -f *.pdf
