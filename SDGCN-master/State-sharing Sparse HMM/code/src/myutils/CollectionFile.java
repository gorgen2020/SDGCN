package myutils;

import java.util.*;
import java.io.*;

//This class handles the file I/O of Collection typed objects.
public class CollectionFile<Type> {
	String filePath = null;

	public CollectionFile(String filePath) throws Exception {
		this.filePath = filePath;
	}
	
	public HashSet<Type> read(Class elementClass) throws Exception {
		HashSet<Type> set = new HashSet<Type>();
		readTo(set, elementClass);
		return set;
	}

	public void readTo(Collection collection, Class elementClass) throws Exception {
		File inputFile = new File(filePath);
		InputStreamReader reader = new InputStreamReader(new FileInputStream(inputFile), "UTF-8");
		BufferedReader bufferedReader = new BufferedReader(reader);
		String line = null;
		while ((line = bufferedReader.readLine()) != null) {
			String parts[] = line.split("\t");
			for (String part : parts) {
				if (part.length() == 0) {
					continue;
				}
				if (elementClass == Integer.class) {
					collection.add((Type) Integer.valueOf(part));
				}
				if (elementClass == Double.class) {
					collection.add((Type) Double.valueOf(part));
				}
				if (elementClass == String.class) {
					collection.add((Type) part);
				}
			}
		}
	}
	
	public void writeFrom(Collection collection) throws Exception {
		File outputFile = new File(filePath);
		if (!outputFile.exists()) {
			outputFile.createNewFile();
		}
		OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8");
		BufferedWriter bufferedWriter = new BufferedWriter(writer);
		for (Object element : collection) {
			bufferedWriter.write(element.toString() + "\n");
		}
		bufferedWriter.flush();
		bufferedWriter.close();
	}
}