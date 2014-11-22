import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.*;

public class Main {
	public static void main(String[] arg) {
		InstanceQuery query;
		try {

			String mQuerry = "SELECT artikel.ID_ARTIKEL, artikel.JUDUL, artikel.FULL_TEXT, artikel_kategori_verified.ID_KELAS"
					+ " FROM artikel"
					+ " INNER JOIN artikel_kategori_verified"
					+ " ON artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL;";

			query = new InstanceQuery();
			query.setUsername("nobody");
			query.setPassword("");
			query.setQuery(mQuerry);
			// You can declare that your data set is sparse
			// query.setSparseData(true);
			Instances data = query.retrieveInstances();

			StringToWordVector strToWV = new StringToWordVector();
			strToWV.setInputFormat(data);
			Instances newData = Filter.useFilter(data, strToWV);

//			System.out.println("newData:");
//			for (int i = 0; i < 20; ++i) {
//				System.out.println(newData.instance(i).toString());
//			}
			
			newData.setClassIndex(newData.numAttributes()-1);
//			Classifier mClassifier = new NaiveBayes();
//			Classifier mClassifier = new J48();
//			Classifier mClassifier = new IBk();
			System.out.println("-------------------Mulai learning nih bro-------------------");
			Classifier mClassifier = new MultilayerPerceptron();
			mClassifier.buildClassifier(newData);
			System.out.println(mClassifier.toString());
			SerializationHelper.write("hasil.model", mClassifier);
			System.out.println("\n-------------------Jalan kok bro-------------------");
			
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		
//		  String a = "weather.nominal.arff";
//		  try{ 
//			  Learning l = new Learning(a,4); 
//			  l.crossValidation(a);
//			  l.fullTraining(a); 
//			  l.classify(); 
//			  l.save("hasil.model");
//		  
//		  } catch(Exception e){ e.printStackTrace(); }
		 
	}
}
