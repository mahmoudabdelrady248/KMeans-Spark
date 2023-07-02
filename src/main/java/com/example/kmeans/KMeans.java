package com.example.kmeans;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.time.Duration;
import java.time.Instant;
import java.lang.Iterable;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;

import scala.Tuple2;

public final class KMeans {
	public static void main(String[] args) throws Exception {
		String inputFile = args[0], outputFile = args[1];int k = Integer.parseInt(args[2]), dim=Integer.parseInt(args[3]);

		JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local").setAppName("kmeans"));
		
		JavaRDD<String> input = sc.textFile(inputFile);
		
		JavaRDD<Vector>data= input.map(new Function<String,Vector>() {
			@Override
			public Vector call(String line) throws Exception {
				String[]s=line.split(",");
				double[]data=new double[dim];
				for(int i=0;i<dim;i++) data[i]=Double.parseDouble(s[i]);
				return new Vector(data);
			}
		});
		
		sc.parallelize(kMeans(data, k, dim)).saveAsTextFile(outputFile+"_"+k+"_"+dim);
	}

	public static List<Vector> kMeans(JavaRDD<Vector> data, int k,int dim) {
		List<Vector>oldCentroids=new ArrayList<>(data.collect().subList(0,k));int iterations=0;
		
		Instant start = Instant.now();
		while(true) {
			JavaPairRDD<Integer, Vector>assignment= data.mapToPair(new PairFunction<Vector, Integer, Vector>() {
				@Override
				public Tuple2<Integer, Vector> call(Vector data) throws Exception {
					int minIndex=-1;double minDistance=Double.POSITIVE_INFINITY;
					for(int i=0;i<k;i++) {
						Double distance=data.squaredDist(oldCentroids.get(i));
						if(distance<minDistance) {minDistance=distance;minIndex=i;}
					}
					return new Tuple2<Integer, Vector>(minIndex,data);
				}
			});
			

			Map<Integer,Vector>newCentroids=assignment.groupByKey().mapValues(new Function<Iterable<Vector>, Vector>() {
				@Override
				public Vector call(Iterable<Vector> elements) throws Exception {
					Vector sum=new Vector(new double[dim]);int size=0;
					for(Vector element:elements) {sum.addInPlace(element);size++;}
					return sum.divide(size);
				}
			}).collectAsMap();
			
			iterations++;
			
			double distance=0.0;
			for(int i=0;i<k;i++) distance+=oldCentroids.get(i).squaredDist(newCentroids.get(i));
			if(distance==0) break;
			
			for(Map.Entry<Integer, Vector> entry:newCentroids.entrySet()) oldCentroids.set(entry.getKey(), entry.getValue());
			
		}
		System.out.println("Time taken : " + Duration.between(start, Instant.now()).toSeconds() +" sec");
		
		System.out.println("Number of iterations : "+iterations);
		
		System.out.println("Final Centroids : ");
		for(Vector centroid:oldCentroids) System.out.println(centroid);
		return oldCentroids;
	}
}
