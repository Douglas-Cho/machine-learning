# machine-learning
This is an external repository of my machine learning work. 

I focus on audit, cybersecurity and financial market related application of machine learning. 

Just as a precaution, you may have to revise the code to make it run without errors as I sanitized the codes not to reveal any piece of  my workplace specific information. Some part of the codes will be also environment specific such as loading data from MSSQL database. 

I hope this helps you see my idea. 

Thank you for reading.  


- Audit_Action_LSTM.py:

This code tests audit trails to screen exceptional transactions. Using labled data with series of users' system activities as sequential inputs, LSTM model will learn and later classify if a new transaction is an exception or not based on the transaction's workflow audit trail. 

For example, a transaction may go through maker's drafting, temporary saving, multiple editing, request for approval, reviewers' approval and go live. These logs of state transition will be converted into a sequential input per transaction for LSTM's learning and classification. 

This code aimed to locate exceptions by analyzing transaction's log but the same concept can be applied to user's activity logs to identify suspicious users. (To do - applying this to UNIX administrators' bash_history) 

- Clustering MNIST data with deep autoencoder and K-means.ipynb:

Clustering encoded cores from autoencoder into n-number of clusters using K-means (To do - for risk assessment clustering) 

- Reinforcement Learning eBest Trading_4actions-version 1.ipynb

I revised a cart-pole DQN code (by Prof Sung Kim) to read in my KOSPI market data as the environment state value. The agent supposed to learn the way to maximize the score using DQN. This model failed to learn (at least by the time I stoped) as the reward was delayed. 


Douglas (Dokeun) Cho 
