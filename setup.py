from distutils.core import setup, Extension
from Cython.Build import cythonize

module1 = Extension('wtmf',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    #include_dirs = ['/usr/local/include'],
                    #libraries = ['tcl83'],
                    #library_dirs = ['/usr/local/lib'],
                    sources = ['wtmf.pyx'],)
                    #extra_compile_args=['-fopenmp'],
                    #extra_link_args=['-fopenmp'],)

setup (name = 'wtmf',
       version = '1.0',
       description = 'Numpy implementation of Weiwei Guo\'s Weighted ' \
                     'Textual Matrix Factorization in sklearn framework.',
       author = 'Chris Kedzie',
       author_email = 'kedzie@cs.columbia.edu',
       url = 'https://github.com/kedz/wtmf',
       long_description = '''
Numpy implementation of Weiwei Guo's Weighted Textual Matrix Factorization in 
sklearn framework.
''',
       ext_modules = cythonize([module1]))
