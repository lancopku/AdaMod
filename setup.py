from setuptools import setup

__VERSION__ = '0.0.3'

setup(name='adamod',
      version=__VERSION__,
      description='AdaMod optimization algorithm, build on PyTorch.',
      long_description=open("README.md", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      keywords=['machine learning', 'deep learning'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      url='https://github.com/karrynest/AdaMod',
      author='Jianbang Ding',
      author_email='jianbangding@pku.edu.cn',
      license='Apache',
      packages=['adamod'],
      install_requires=[
          'torch>=0.4.0',
      ],
      zip_safe=False,
      python_requires='>=3.6.0')
