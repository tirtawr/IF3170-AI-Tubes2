import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.*;

public class Learning {
	private int mode; //1 untuk naive bayes, 2 untuk tree, 3 untuk KNN, 4 untuk ANN
	private Instances data;
	private Instances unLabeledData;
	private Classifier classifier;
	
	Learning(String filename,int m) throws Exception{
		
		data=DataSource.read(filename);
		mode=m;
		classifier = null;
	}
	
	
	public Instances readData(String file) throws Exception{
		Instances dataRead = null;
		try{
			dataRead = DataSource.read(file);
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return dataRead;
	}
	
	public void crossValidation(String file) throws Exception{
		Instances train = readData(file);
		
		train.setClassIndex(train.numAttributes()-1);
		train.stratify(10);
		
		if(mode==1){
			classifier = (Classifier) new NaiveBayes();
		}
		else if(mode==2){
			classifier = (Classifier) new J48();
		}
		else if(mode==3){
			classifier = (Classifier) new IBk();
		}
		else if(mode==4){
			classifier = (Classifier) new MultilayerPerceptron();
		}
		classifier.buildClassifier(train);
		
		System.out.println(classifier.toString());
		
		data.setClassIndex(data.numAttributes()-1);
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));
		
		System.out.println(eval.toSummaryString("summary",false));
		System.out.println(eval.toClassDetailsString("Detail"));
		System.out.println(eval.toMatrixString());
	}
	
	public void fullTraining(String file) throws Exception{
		data = readData(file);
		data.setClassIndex(data.numAttributes()-1);
		
		if(mode==1){
			classifier = (Classifier) new NaiveBayes();
		}
		else if(mode==2){
			classifier = (Classifier) new J48();
		}
		else if(mode==3){
			classifier = (Classifier) new IBk();
		}
		else if(mode==4){
			classifier = (Classifier) new MultilayerPerceptron();
		}
		classifier.buildClassifier(data);
		
		System.out.println(classifier.toString());
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(classifier, data);
		
		System.out.println(eval.toSummaryString("summary",false));
		System.out.println(eval.toClassDetailsString("Detail"));
		System.out.println(eval.toMatrixString());
	}
	
	public void classify() throws Exception{
		unLabeledData = DataSource.read("unlabeled.arff");
		unLabeledData.setClassIndex(unLabeledData.numAttributes()-1);
		Instances LabeledData = new Instances(unLabeledData);

		for(int i=0; i < unLabeledData.numInstances();++i){
			double clsLabel = classifier.classifyInstance(unLabeledData.instance(i));
			LabeledData.instance(i).setClassValue(clsLabel);
		}
		System.out.println(LabeledData.toString());
	}
	
	
	public void save(String filename) throws Exception{
		SerializationHelper.write(filename, classifier);
	}
	
	public void load(String filename) throws Exception{
		classifier = (Classifier) SerializationHelper.read(filename);
	}
	
	
	
}
