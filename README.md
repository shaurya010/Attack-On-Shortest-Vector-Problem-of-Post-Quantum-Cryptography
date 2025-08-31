<!DOCTYPE html>
<html>
<head>
  <title>Attack on Lattice SVP using KNN</title>
</head>
<body>
  <h1>🔐 Attack on Lattice Shortest Vector Problem using K-Nearest Neighbour</h1>
  <p>
    This repository contains research work and implementation details from the paper 
    <b>"Attack on Lattice Shortest Vector Problem using K-Nearest Neighbour"</b>.
  </p>

  <h2>📖 Abstract</h2>
  <p>
    Lattice-based cryptography is one of the most secure and adaptable branches of post-quantum cryptography. 
    The <b>Shortest Vector Problem (SVP)</b> forms the foundation of many lattice-based cryptosystems. 
    In this project, we explore attacks on SVP (2D, 4D, and 10D lattices) using the <b>K-Nearest Neighbour (KNN)</b> 
    machine learning algorithm. Our results demonstrate accuracy of up to <b>80% on 2D</b> and <b>65% on 10D</b> lattices 
    using self-prepared datasets.
  </p>

  <h2>🛠 Methodology</h2>
  <ul>
    <li>Generated lattice datasets (2D, 4D, 10D) for experimentation.</li>
    <li>Applied <b>KNN classification</b> using Euclidean distance to identify shortest non-zero vectors.</li>
    <li>Performed iterative reduction of lattice size to improve attack accuracy.</li>
    <li>Compared performance against other ML approaches such as K-Means.</li>
  </ul>

  <h2>📊 Results</h2>
  <ul>
    <li>2D Lattice → Accuracy up to <b>78%</b></li>
    <li>4D Lattice → Accuracy up to <b>61%</b></li>
    <li>10D Lattice → Accuracy up to <b>60%</b></li>
  </ul>

  <h2>📌 Keywords</h2>
  <p>
    Shortest Vector Problem (SVP), Lattice-based Cryptography (LBC), 
    K-Nearest Neighbour (KNN), Post-Quantum Cryptography (PQC), Euclidean Distance.
  </p>

  <h2>👥 Authors</h2>
  <ul>
    <li>Shaurya Pratap Singh</li>
    <li>Brijesh Kumar Chaurasia</li>
    <li>Tanmay Tripathi</li>
    <li>Ayush Pal</li>
    <li>Siddharth Gupta</li>
  </ul>

  <h2>📌 Citation</h2>
  <p>
    If you use this work, please cite:<br>
    <i>S. P. Singh, B. K. Chaurasia, T. Tripathi, A. Pal, and S. Gupta. 
    "Attack on Lattice Shortest Vector Problem using K-Nearest Neighbour."</i>
  </p>

  <h2>🚀 Future Work</h2>
  <p>
    Future research directions include testing hybrid models and exploring other ML approaches 
    beyond KNN to improve attack success rates on higher-dimensional lattices.
  </p>
</body>
</html>
