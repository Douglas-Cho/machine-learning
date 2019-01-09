# machine-learning
This is an external repository of my machine learning work. I focus on audit, cybersecurity and financial market related applications of machine learning. 

Just as a precaution, you may have to revise the codes to make them run without errors as I sanitized the codes not to reveal any parts of my workplace specific information. Some parts of the codes will also require specific environment to work such as loading data from MS-SQL database. I just hope you see my idea in that case. 

- Audit_Action_LSTM.py:

This code tests audit trails to detect potentially abnormal transactions. Using series of users' system activities from the workflow audit trails, the LSTM model will learn and later classify if a new transaction is a potential exception or not. 

For example, a transaction may go through maker's drafting, temporary saving, multiple editing, requesting for approval, and then go live after reviewers' approval. These logs of state transition will be converted into a sequential input per transaction for LSTM's learning and classification. 

This code can be applied to user's activity logs to detect suspicious activities. (To do - applying this to UNIX administrators' bash_history) 

- Clustering MNIST data with deep autoencoder and K-means.ipynb:

This code clusters the encoded cores from autoencoder into n-number of clusters using K-means. 

- Reinforcement Learning eBest Trading_4actions-version 1.ipynb

I revised a cart-pole Deep Q-Network code (by Prof Sung Kim) to read in my KOSPI market data as the environment state values. The agent supposed to learn the way to maximize the score using DQN. This model failed to learn (at least by the time I stoped) as the reward was delayed. 

- character-based-lstm2.py:

This code originally came from Kaggle's Toxic Comment Classification Challenge. This embedds letters not the words. This means all characters will be tokenized and used as input streams. I just edited the code here and there to target one label and produce confusion matrix. This character based model may fit well for the sanction list fuzzy matching purpose as well. 

- reinforcement-learning:

I revised the cartpole DQN example from the rlcode github to create a DRQN example. I also expanded it with doubling, dueling, prioritized experience replay and LSTM burn-in. 




Douglas (Dokeun) Cho 
