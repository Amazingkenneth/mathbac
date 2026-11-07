# Task: reproduce Cha et al. 2022

Reproduce the computational results of:

> Cha, M., Turali Emre, E. S., Xiao, X., Kim, J.-Y., Bogdan, P., VanEpps, J. S., Violi, A. & Kotov, N. A. Unifying structural descriptors for biological and bioinspired nanoscale complexes. *Nat. Comput. Sci.* **2**, 243–252 (2022). doi:10.1038/s43588-022-00229-w

The paper computes chemical (CH), geometrical (GE) and graph-theoretical (GT) descriptors for amino-acid residues in protein–protein complexes, trains classifiers to predict which residue pairs are in close contact, and then applies the models trained on protein–protein interactions to complexes of proteins with nanoparticles (graphene quantum dots, carbon nanotubes, spherical carbon nanoparticles, ZnO nanopyramids).

The reproduction is complete when the workspace contains, following the artifact contracts:

1. A method specification extracted from the paper and supplementary information (`spec/spec.yaml`).
2. A working environment, with a smoke-test report for every tool used (`env/smoke_report.json`).
3. The protein–protein training set and the held-out complexes: pair lists, Cα coordinates, distance matrices and contact labels (`data/`).
4. Coarse-grained nanoparticle structures and the list of protein–nanoparticle cases (`data/np/`).
5. Per-residue CH, GE and GT descriptors for every chain, and GE+GT descriptors for every nanoparticle (`features/`).
6. Cross-validated results for every model on every descriptor subset, with the raw per-fold predictions (`ml/cv/`).
7. Interface predictions for the held-out protein–protein complexes (`predictions/ppi/`).
8. Interface predictions for the protein–nanoparticle cases and the protein-corona ranking (`predictions/pni/`).
9. A reproduction report with a verdict for each of the paper's main claims, and tool-card drafts for each validated step (`report/`).

Claims the report must address:

- GT descriptors alone predict residue contacts better than GE alone, and GE better than CH alone.
- Adding CH descriptors to GE or GT changes performance by only about two points or less.
- Random forest, XGBoost and the neural network perform similarly, and much better than logistic regression, naive Bayes and SVM.
- GE+GT models predict the interfaces of held-out, non-homologous protein complexes with about 80% accuracy.
- Models trained on protein–protein complexes, using only GE+GT descriptors, predict nanoparticle–protein interaction sites that match experiments and simulations reported in the literature.

Compute is limited. Heavy steps go through the compute queue, and any subsampling must be documented.
