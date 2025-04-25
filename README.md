# CDC-Calculator

Author: Caleb Spink, B.S. 

Contact: spinkc@kennedykrieger.org

License: Some sort of copyleft

This is a calculator and graphing tool utilizing the conservative dual-criterion method (CDC) where trend and level lines are increased or decreased by 0.25x the standard deviation from the previous phase. The number of points below and above both lines are tested for statistical significance (Fisher et al., 2003). However, the program uses Fisher et al.'s (2003) modified list to for comparison with 3/3 and 4/4 added.

The GUI was made using Custom TKinter (https://github.com/TomSchimansky/CustomTkinter.git). 

Task Analysis:

1. Download the program and open it
2. Enter information. "# of Phases" and "# of Sessions" are the only necessary fields and must contain integers.
3. Click "Generate Table"
4. There are two dropdown menus per phase. One is for condition name selection and the other is for selecting predicted direction ("+" for increasing and "-" for decreasing).
5. Click "Generate Graph" and interpretations will appear in the output console. A seperate window featuring the graph with appear.
6. Click "Save Graph"
7. Click "Clear All" to, well, clear all

References: 

Fisher W. W., Kelley, M. E., & Lomas, J. E. (2003). Visual aids and structured criteria for improving visual inspection and interpretation of single-case designs. Journal of Applied Behavior Analysis, 36, 387-406. https://doi.org/10.1901/jaba.2003.36-387 
