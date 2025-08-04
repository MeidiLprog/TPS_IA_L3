
#LEFKI MEIDI / Boukraa Asma
#L3 Informatique
#Groupe TD-2
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling


# ---------Create nodes and conditional probabilityh distribution ------

def get_probs_gene_ancestor(varName):
    """
    Builds a conditional probability distribution (TabularCPD)
    using a priori probabilities, for a variable gene with no parents.
    """
    return TabularCPD(
        variable=varName,
        variable_card=3,
        values=[
            [0.01],  # P(Gene=2)
            [0.03],  # P(Gene=1)
            [0.96]   # P(Gene=0)
        ],
        state_names={varName: [2, 1, 0]}
    )



def get_probs_trait(varName, evidenceName):
    """
    Builds a conditional probability distribution (TabularCPD)
    for traits given the number of genes.
    """
    return TabularCPD(
        variable=varName,
        variable_card=2,  # Trait peut être 'oui' ou 'non'
        values=[
            [0.65, 0.56, 0.01],  # P(Trait=oui|Gene=2,1,0)
            [0.35, 0.44, 0.99]   # P(Trait=non|Gene=2,1,0)
        ],
        evidence=[evidenceName],
        evidence_card=[3],
        state_names={
            varName: ["vrai", "faux"],
            evidenceName: [2, 1, 0]
        }
    )


# constant defining mutation probability of a gene
prob_mutation = 0.01

def get_probs_heredity1(geneParent):
    """
    Computes probability of inheriting 1 gene from
    a given parent: 
    P(Gene_{inherited chromosome}|Gene_{parent})

    Parameter
    ----------
    geneParent: number of genes (0, 1 or 2) of the
    parent (father or mother)
    """


    if geneParent == 1:

        return 0.5 * (1-prob_mutation) + 0.5*prob_mutation

    elif geneParent == 2:
        return 1.0*(1-prob_mutation)
    else: 
        return prob_mutation



def get_probs_gene(varNameChild,evidenceNameFather,evidenceNameMother):
    """
    Builds a conditional probability distribution (TabularCPD)
    for the number of genes of a child given the number of genes
    of each of the parents

    Parameters
    ----------
    varNameChild : name of the traits variable (String)
    evidenceName: name of the evidence gene variable (String)
    """
    probabilites=[[],[],[]]

    for mere in range(3):
        for pere in range(3):

            prob_pere = get_probs_heredity1(pere)
            prob_mere = get_probs_heredity1(mere)

           
            proba_0 = (1-prob_mere)*(1-prob_pere)
        
            proba_1 = (1-prob_pere)*prob_mere + (1-prob_mere)*prob_pere

            proba_2 = prob_pere*prob_mere
            

            probabilites[0].append(proba_0)
            probabilites[1].append(proba_1)
            probabilites[2].append(proba_2)

 

    return TabularCPD (
        variable=varNameChild,
        variable_card=3,
        values=probabilites,
        evidence=[evidenceNameFather,evidenceNameMother],
        evidence_card=[3,3],
        state_names={varNameChild:[2,1,0],
                     evidenceNameFather:[2,1,0],
                     evidenceNameMother:[2,1,0]
            
                     }

    )


print(get_probs_gene_ancestor("test"))
print(get_probs_trait("test","Gene"))
print()

cpd_gene_ancestor = get_probs_gene_ancestor("Gene_Ancestor")
print("CPD pour le gène ancêtre :")
print(cpd_gene_ancestor)
print()

# 2. Obtenez les CPDs pour les traits en fonction du nombre de gènes
cpd_trait_father = get_probs_trait("Trait_Father", "Gene_Father")
print("CPD pour le trait du père :")
print(cpd_trait_father)
print()

cpd_trait_mother = get_probs_trait("Trait_Mother", "Gene_Mother")
print("CPD pour le trait de la mère :")
print(cpd_trait_mother)
print()

# 3. Obtenez les CPDs pour l'hérédité des gènes de l'enfant
cpd_gene = get_probs_gene("Gene_Child", "Gene_Father", "Gene_Mother")
print("CPD pour l'hérédité des gènes de l'enfant :")
print(cpd_gene)
print()


#CPD pour l'hérédité des gènes de l'enfant :
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+
#| Gene_Father   | Gene_Father(2) | Gene_Father(2) | Gene_Father(2)      | Gene_Father(1) | Gene_Father(1) | Gene_Father(1)        | Gene_Father(0)      | Gene_Father(0)        | Gene_Father(0)         |
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+
#| Gene_Mother   | Gene_Mother(2) | Gene_Mother(1) | Gene_Mother(0)      | Gene_Mother(2) | Gene_Mother(1) | Gene_Mother(0)        | Gene_Mother(2)      | Gene_Mother(1)        | Gene_Mother(0)         |
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+
#| Gene_Child(2) | 0.9801         | 0.495          | 0.00990000000000001 | 0.495          | 0.25           | 0.0050000000000000044 | 0.00990000000000001 | 0.0050000000000000044 | 0.00010000000000000018 |
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+
#| Gene_Child(1) | 0.0198         | 0.5            | 0.9802              | 0.5            | 0.5            | 0.5                   | 0.9802              | 0.5                   | 0.01980000000000002    |
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+
#| Gene_Child(0) | 0.0001         | 0.005          | 0.0099              | 0.005          | 0.25           | 0.495                 | 0.0099              | 0.495                 | 0.9801                 |
#+---------------+----------------+----------------+---------------------+----------------+----------------+-----------------------+---------------------+-----------------------+------------------------+



# -------------------- Bayesian Network for Family 1 --------------------
model1 = BayesianNetwork([
    ('G_Leto', 'T_Leto'),       
    ('G_Jessica', 'T_Jessica'), 
    ('G_Leto', 'G_Paul'),       
    ('G_Jessica', 'G_Paul'),    
    ('G_Leto', 'G_Alia'),       
    ('G_Jessica', 'G_Alia'),    
    ('G_Paul', 'T_Paul'),       
    ('G_Alia', 'T_Alia')        
])

# Define CPDs for Family 1
cpd_G_Leto = get_probs_gene_ancestor('G_Leto')
cpd_G_Jessica = get_probs_gene_ancestor('G_Jessica')
cpd_T_Leto = get_probs_trait('T_Leto', 'G_Leto')
cpd_T_Jessica = get_probs_trait('T_Jessica', 'G_Jessica')
cpd_G_Paul = get_probs_gene('G_Paul', 'G_Leto', 'G_Jessica')
cpd_T_Paul = get_probs_trait('T_Paul', 'G_Paul')
cpd_G_Alia = get_probs_gene('G_Alia', 'G_Leto', 'G_Jessica')
cpd_T_Alia = get_probs_trait('T_Alia', 'G_Alia')

model1.add_cpds(cpd_G_Leto, cpd_G_Jessica, cpd_T_Leto, cpd_T_Jessica, cpd_G_Paul, cpd_T_Paul, cpd_G_Alia, cpd_T_Alia)
model1.check_model()

viz = model1.to_graphviz()
viz.draw('family1.png', prog='dot')

# -------------------- Exact Inference for Family 1 --------------------
infer_family1 = VariableElimination(model1)

# Define all observed T values for Family 1 based on the diagram
observed_T_values = {
    'T_Leto': 'faux',
    'T_Jessica': 'vrai',
    'T_Paul': 'faux',
    'T_Alia': 'vrai'
}

# Perform exact inference for G_Paul given all observed T values
g_dist_paul_all_T = infer_family1.query(variables=['G_Paul'], evidence=observed_T_values)
print("P(G_Paul | Toutes les valeurs observées de T) :", g_dist_paul_all_T)
# Perform exact inference for G_Paul given all observed T values and additional gene information for Jessica and Alia
observed_T_values_with_genes = observed_T_values.copy()
observed_T_values_with_genes.update({'G_Jessica': 1, 'G_Alia': 2})

g_dist_paul_jessica_alia = infer_family1.query(
    variables=['G_Paul'],
    evidence=observed_T_values_with_genes
)
print("P(G_Paul | Toutes les valeurs observées de T, G_Jessica=1, G_Alia=2) :", g_dist_paul_jessica_alia)
# -------------------- Approximate Inference Using Sampling --------------------
# Forward Sampling to estimate distribution of G_Paul given all T values
sampler = BayesianModelSampling(model1)
data = sampler.forward_sample(size=10000)

# Filter samples where all observed T values are met
filtered_samples = data[
    (data['T_Leto'] == 'faux') &
    (data['T_Jessica'] == 'vrai') &
    (data['T_Paul'] == 'faux') &
    (data['T_Alia'] == 'vrai')
]
approx_dist_paul = filtered_samples['G_Paul'].value_counts(normalize=True)
print("Distribution approximative de G_Paul étant donné toutes les valeurs de T :")
print(approx_dist_paul)

# Rejection Sampling for P(G_Paul | All T values, G_Jessica=1, G_Alia=2)
evidence = [
    State(var='T_Leto', state='faux'),
    State(var='T_Jessica', state='vrai'),
    State(var='T_Paul', state='faux'),
    State(var='T_Alia', state='vrai'),
    State(var='G_Jessica', state=1),
    State(var='G_Alia', state=2)
]
samples = sampler.rejection_sample(evidence=evidence, size=100)
approx_dist_paul_rejection = samples['G_Paul'].value_counts(normalize=True)
print("Distribution approximative de G_Paul étant donné toutes les valeurs de T et G_Jessica=1, G_Alia=2 (Échantillonnage par rejet) :")
print(approx_dist_paul_rejection)
# -------------------- Bayesian Network for Family 2 --------------------
model2 = BayesianNetwork([
    ('G_Charles', 'T_Charles'),
    ('G_Diana', 'T_Diana'),
    ('G_Michael', 'T_Michael'),
    ('G_Carole', 'T_Carole'),
    ('G_Meghan', 'T_Meghan'),
    ('G_Harry', 'T_Harry'),
    ('G_William', 'T_William'),
    ('G_Katherine', 'T_Katherine'),
    ('G_Philippa', 'T_Philippa'),
    ('G_Charles', 'G_Harry'),
    ('G_Diana', 'G_Harry'),
    ('G_Charles', 'G_William'),
    ('G_Diana', 'G_William'),
    ('G_Michael', 'G_Katherine'),
    ('G_Carole', 'G_Katherine'),
    ('G_Michael', 'G_Philippa'),
    ('G_Carole', 'G_Philippa'),
    ('G_Harry', 'G_Archie'),
    ('G_Meghan', 'G_Archie'),
    ('G_Harry', 'G_Liliet'),
    ('G_Meghan', 'G_Liliet'),
    ('G_William', 'G_George'),
    ('G_Katherine', 'G_George'),
    ('G_William', 'G_Charlotte'),
    ('G_Katherine', 'G_Charlotte'),
    ('G_William', 'G_Louis'),
    ('G_Katherine', 'G_Louis'),
    
    # Add trait nodes for third-generation members
    ('G_Archie', 'T_Archie'),
    ('G_Liliet', 'T_Liliet'),
    ('G_George', 'T_George'),
    ('G_Charlotte', 'T_Charlotte'),
    ('G_Louis', 'T_Louis')
])

# Define CPDs for Family 2
# Ancestor nodes
cpd_G_Charles = get_probs_gene_ancestor('G_Charles')
cpd_G_Diana = get_probs_gene_ancestor('G_Diana')
cpd_G_Michael = get_probs_gene_ancestor('G_Michael')
cpd_G_Carole = get_probs_gene_ancestor('G_Carole')
cpd_G_Meghan = get_probs_gene_ancestor('G_Meghan')

# Trait CPDs for ancestors
cpd_T_Charles = get_probs_trait('T_Charles', 'G_Charles')
cpd_T_Diana = get_probs_trait('T_Diana', 'G_Diana')
cpd_T_Michael = get_probs_trait('T_Michael', 'G_Michael')
cpd_T_Carole = get_probs_trait('T_Carole', 'G_Carole')
cpd_T_Meghan = get_probs_trait('T_Meghan', 'G_Meghan')

# Gene inheritance CPDs
cpd_G_Harry = get_probs_gene('G_Harry', 'G_Charles', 'G_Diana')
cpd_G_William = get_probs_gene('G_William', 'G_Charles', 'G_Diana')
cpd_G_Katherine = get_probs_gene('G_Katherine', 'G_Michael', 'G_Carole')
cpd_G_Philippa = get_probs_gene('G_Philippa', 'G_Michael', 'G_Carole')

# Trait CPDs for children
cpd_T_Harry = get_probs_trait('T_Harry', 'G_Harry')
cpd_T_William = get_probs_trait('T_William', 'G_William')
cpd_T_Katherine = get_probs_trait('T_Katherine', 'G_Katherine')
cpd_T_Philippa = get_probs_trait('T_Philippa', 'G_Philippa')

# Define third-generation gene and trait CPDs
cpd_G_Archie = get_probs_gene('G_Archie', 'G_Harry', 'G_Meghan')
cpd_T_Archie = get_probs_trait('T_Archie', 'G_Archie')
cpd_G_Liliet = get_probs_gene('G_Liliet', 'G_Harry', 'G_Meghan')
cpd_T_Liliet = get_probs_trait('T_Liliet', 'G_Liliet')
cpd_G_George = get_probs_gene('G_George', 'G_William', 'G_Katherine')
cpd_T_George = get_probs_trait('T_George', 'G_George')
cpd_G_Charlotte = get_probs_gene('G_Charlotte', 'G_William', 'G_Katherine')
cpd_T_Charlotte = get_probs_trait('T_Charlotte', 'G_Charlotte')
cpd_G_Louis = get_probs_gene('G_Louis', 'G_William', 'G_Katherine')
cpd_T_Louis = get_probs_trait('T_Louis', 'G_Louis')

# Add CPDs to the model
model2.add_cpds(
    cpd_G_Charles, cpd_G_Diana, cpd_G_Michael, cpd_G_Carole, cpd_G_Meghan,
    cpd_T_Charles, cpd_T_Diana, cpd_T_Michael, cpd_T_Carole, cpd_T_Meghan,
    cpd_G_Harry, cpd_G_William, cpd_G_Katherine, cpd_G_Philippa,
    cpd_T_Harry, cpd_T_William, cpd_T_Katherine, cpd_T_Philippa,
    cpd_G_Archie, cpd_T_Archie, cpd_G_Liliet, cpd_T_Liliet,
    cpd_G_George, cpd_T_George, cpd_G_Charlotte, cpd_T_Charlotte, cpd_G_Louis, cpd_T_Louis
)

model2.check_model()

viz = model2.to_graphviz()
viz.draw('family2.png', prog='dot')

# -------------------- Exact Inference for Family 2 --------------------
infer_family2 = VariableElimination(model2)

# Prediction of G for George and Liliet, knowing the observed T values
evidence_T = {
    'T_Charles': 'vrai', 'T_Diana': 'faux', 'T_Michael': 'faux', 'T_Carole': 'faux',
    'T_Harry': 'faux', 'T_Meghan': 'faux', 'T_William': 'faux', 'T_Katherine': 'faux',
    'T_Philippa': 'vrai', 'T_Archie': 'vrai', 'T_Liliet': 'faux', 'T_George': 'faux',
    'T_Charlotte': 'faux', 'T_Louis': 'vrai'
}
result_George = infer_family2.query(variables=['G_George'], evidence=evidence_T)
result_Liliet = infer_family2.query(variables=['G_Liliet'], evidence=evidence_T)
print("P(G_George | Valeurs observées de T) :", result_George)
print("P(G_Liliet | Valeurs observées de T) :", result_Liliet)

# Prediction of G for George and Liliet, knowing that Katherine has no defective gene
evidence_T_Katherine = evidence_T.copy()
evidence_T_Katherine.update({'G_Katherine': 0})
result_George_Katherine = infer_family2.query(variables=['G_George'], evidence=evidence_T_Katherine)
result_Liliet_Katherine = infer_family2.query(variables=['G_Liliet'], evidence=evidence_T_Katherine)
print("P(G_George | Valeurs observées de T, G_Katherine=0) :", result_George_Katherine)
print("P(G_Liliet | Valeurs observées de T, G_Katherine=0) :", result_Liliet_Katherine)

