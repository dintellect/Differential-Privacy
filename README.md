# Differential-Privacy

**Differential Privacy implemented on the Recommender's System**

***INTRODUCTION***

A recommender system attempts to predict a user’s potential likes and interests
by analyzing the user’s historical transaction data. Currently recommender systems
are highly successful on e-commerce web sites capable of recommending
products users will probably like. Collaborative Filtering (CF) is one of the most
popular recommendation techniques as it is insensitive to product details. This is
achieved by analyzing the user’s historical transaction data with various data mining
or machine learning techniques, e.g. k nearest neighbor rule, the probability theory
and matrix factorization.

***OBJECTIVE***

The aim of the project was to solve the problem of inferring an individual’s rating, especially for the neighborhood-based methods using some background information of the individual. For example, an adversary can infer the rating history of an active user by creating fake neighbors based on background information.

***EARLIER PRIVACY APPROACHES AND THEIR DRAWBACKS***

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

***DIFFERENTIAL PRIVACY***

Differential privacy was introduced into CF by McSherry et al., who
pioneered a study that constructed the private covariance matrix to randomize each
user’s rating before submitting to the system.Machanavajjhala et al presented
a graph link-based recommendation algorithm and formalized the trade-off between
accuracy and privacy. We will applying this approach in our recoomender system. 

***RESOLUTION***

The process of differential privacy solves this problem by using the fact that user’s rating should be inferred from the
entire database of users by weighing each user with their “similarity score” with the reference user. Thus user’s
rating comes from the entire population rather than some predefined set of users. This has two advantages – first
we can never identify the ratings of that individual user, second, it is computationally very effective.

***OVERVIEW***

Application             : Recommender systems

Input data              : User-item rating matrix

Output data             : Prediction

Challenges              : High sensitivity

Solutions               : Adjust sensitivity measurement

Selected mechanism      : Group large candidate set
