# Differential-Privacy

Differential Privacy implemented on the Recommender's System

----INTRODUCTION------

A recommender system attempts to predict a user’s potential likes and interests
by analyzing the user’s historical transaction data. Currently recommender systems
are highly successful on e-commerce web sites capable of recommending
products users will probably like. Collaborative Filtering (CF) is one of the most
popular recommendation techniques as it is insensitive to product details. This is
achieved by analyzing the user’s historical transaction data with various data mining
or machine learning techniques, e.g. k nearest neighbor rule, the probability theory
and matrix factorization.

-----PROBLEM STATEMENT-------

The literature has shown that continual observation of recommendations with
some background information makes it possible to infer the user details. For
example, an adversary can infer the rating history of an active user by creating fake
neighbors based on background information.

-------EARLIER PRIVACY APPROACHES AND THEIR DRAWBACKS------

A collaborative filtering method employs certain traditional privacy
preserving approaches, such as cryptographic, obfuscation and perturbation.Among
them, Cryptographic is suitable for multiple parties but induces extra computational
cost. Obfuscation is easy to understand and implement, however the utility
will decrease significantly. Perturbation preserves high privacy levels by
adding noise to the original dataset, but the magnitude of noise is subjective and
hard to control. Moreover, these traditional approaches suffer from a common
weakness: the privacy notion is weak and hard to prove theoretically, thus impairing
the credibility of the final result. In order to address these problems, differential
privacy has been proposed.

--------DIFFERENTIAL PRIVACY------

Differential privacy was introduced into CF by McSherry et al., who
pioneered a study that constructed the private covariance matrix to randomize each
user’s rating before submitting to the system.Machanavajjhala et al presented
a graph link-based recommendation algorithm and formalized the trade-off between
accuracy and privacy. We will applying this approach in our recoomender system. 

---------OVERVIEW-----------

Application             : Recommender systems
Input data              : User-item rating matrix
Output data             : Prediction
Challenges              : High sensitivity
Solutions               : Adjust sensitivity measurement
Selected mechanism      : Group large candidate set
