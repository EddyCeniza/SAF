SAF: Sentiment Analysis for Films

	Gregory Khrom-Abramyan
	Philip Phan
	Eddy Ceniza



Libraries and Tools Used:

	pytorch
	pandas
	sklearn

	spaCy
	SpacyTextBlob
	transformers
	


Training Models:

	1. Move "Data" folder into desired model directory (spaCy or DistilBERT)
	2. Run the "Main.py" file within the chosen model directory (By default, the 10,000 point dataset is loaded)

Running Pre-Trained DistilBERT Model:

	1. Open "Prediction.py" inside the "DistilBERT" Folder
	2. Replace text on line 25 with test input
	3. Run "Prediciton.py"

Interpreting Results:

	The DistilBERT model prints a 2-tuple of the model's prediction confidence of the form:
	
		{(Negative prediction confidence), (Positive prediction confidence)}

	The confidence is on a scale of 0 to 1. If Negative/Positive Confidence is > 0.5, the input is classified as having negative/positive sentiment accordingly.

