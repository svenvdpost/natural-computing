import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;

import javax.print.event.PrintJobListener;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import util.Debug;
import util.Settings;
import algorithms.ContiguousCountingDAG;
import algorithms.PatternTrie;
import algorithms.RChunkPatterns;
import alphabets.Alphabet;
import alphabets.AminoAcidAlphabet;
import alphabets.BinaryAlphabet;
import alphabets.BinaryLetterAlphabet;
import alphabets.DegenerateAminoAcidAlphabet;
import alphabets.LatinAlphabet;

import java.io.PrintWriter;

/*
 * Created on 27.08.2009 by Johannes Textor
 * This Code is licensed under the BSD license:
 * http://www.opensource.org/licenses/bsd-license.php
 */

public class Main {

	// File englishtest = new File(path + "english.test");
	// File tagalogtest = new File(path + "tagalog.test");

	// File hiligaynon = new File(path+"lang\\hiligaynon.txt");
	// File middle_english = new File(path+"lang\\middle-english.txt");
	// File plautdietsch = new File(path+"lang\\plautdietsch.txt");
	// File xhosa = new File(path+"lang\\xhosa.txt");

	public static PrintWriter out;
	public static String path = "C:\\Users\\svenv\\OneDrive\\Documenten\\Uni\\22-23\\Natural Computing\\negative-selection\\";

	// File to run through the classifier
	public static File testset1 = new File(path + "syscalls\\snd-cert\\snd-cert.1.test");
	public static File labelset1 = new File(path + "syscalls\\snd-cert\\snd-cert.1.labels");
	public static File testset2 = new File(path + "syscalls\\snd-cert\\snd-cert.2.test");
	public static File labelset2 = new File(path + "syscalls\\snd-cert\\snd-cert.2.labels");
	public static File testset3 = new File(path + "syscalls\\snd-cert\\snd-cert.3.test");
	public static File labelset3 = new File(path + "syscalls\\snd-cert\\snd-cert.3.labels");

	// array of files to run through the classifier
	public static File[] testfiles = { testset1, testset2, testset3 };
	// array of files containing the labels
	public static File[] labelfiles = { labelset1, labelset2, labelset3 };
	// array of strings containing a brief name for each data set
	public static String[] setname = { "snd-cert.1", "snd-cert.2", "snd-cert.3" };

	// Name of the output file
	public static String outputfile = "snd-cert" + ".csv";
	// public static File inputfile = new File(path +
	// "syscalls\\snd-cert\\snd-cert.2.test");

	public static Vector<String> preprocess(String line, int n, boolean overlapping) {

		Vector<String> linechunks = new Vector<String>();
		String chunk;
		int division = (int) (line.length() / n);

		int leftover = line.length() % n;

		if (overlapping) {
			for (int i = 0; i < line.length() - n; i++) {
				if (i < line.length() - n) {
					chunk = line.substring(i, i + n);
					linechunks.add(chunk);
				}
			}
		} else {

			StringBuilder chunkSB = new StringBuilder();
			// System.out.println("length line: " + line.length());

			for (int i = 0; i < division; i++) {
				// System.out.println("index substring start: " + (i*n) + " substring end: " +
				// (i*n+n-1));
				chunk = line.substring(i * n, i * n + n);
				linechunks.add(chunk);
			}

			if (leftover > 0) {

				// System.out.println((line.length()-leftover) + " " + (line.length()-1));

				String substring = line.substring(line.length() - leftover, line.length());

				for (int i = 0; i < n; i++) {
					// System.out.println("index leftover: " + i % leftover);
					chunkSB.append(substring.charAt(i % leftover));
				}

				chunk = chunkSB.toString();
				// System.out.println(chunk);
				linechunks.add(chunk);
			}

		}

		return linechunks;

	}

	public static void main(String[] args) {
		// "-alphabet", "file://"+path+"english.train",
		// args = new String[] { "-self", path + "english.train", "-n", "10", "-r", "4",
		// "-c", "-l" };

		try {
			// printwriter to write to the output file
			out = new PrintWriter(String.format(outputfile));
			// column names in the output file
			out.println("filename, anomaly_score, label, r, n");

			// for different r values, run the program


			for (int n = 5; n <= 15; n++) {
				for (int r = 2; r <= 4; r++) {
					args = new String[] { "-self", path + "syscalls\\snd-cert\\snd-cert.train", "-n",
							Integer.toString(n), "-r", Integer.toString(r), "-c",
							"-l" };
					run(args);
				}
			}
			// close the output file
			out.close();

		} catch (FileNotFoundException e) {
			System.out.println("FileNotFoundException");
		}
	}

	public static void run(String[] args) throws FileNotFoundException {

		Options myOptions = new Options();
		myOptions.addOption(new Option("n", true,
				"Length of strings in self set"));
		myOptions.addOption(new Option("r", true, "Parameter r <= n"));
		myOptions.addOption(new Option("o", false, "Offset into strings"));
		myOptions.addOption(new Option("d", true, "Add to alphabets the digits from 0 to ..."));
		myOptions.addOption(new Option("g", false, "Print debug information"));
		myOptions.addOption(new Option("v", false, "Invert match (like grep)"));
		// myOptions.addOption(new Option("b", false, "Subtract baseline noise"));
		myOptions.addOption(new Option("c", false, "Count matching detectors instead of binary match"));
		myOptions.addOption(new Option("l", false, "Output logarithms instead of actual values"));
		myOptions.addOption(new Option("k", false, "Use r-chunk instead of r-contiguous matching"));
		myOptions.addOption(new Option("p", true, "Output k-th component of matching profile (0 for full profile)"));
		myOptions.addOption(new Option("self", true,
				"File containing self set (1 string per line)"));
		myOptions.addOption(new Option("alphabet", true,
				"Alphabet, currently one of [infer|binary|binaryletter|amino|damino|latin]. Default: infer (uses all characters from \"self\" file as alphabet). Alternatively, specify file://[f] to set the alphabet to all characters found in file [f]."));

		CommandLineParser parser = new BasicParser();

		int n = 0;
		int r = 0; // matching used for training
		int r2 = 0; // matching used for classification
		String self = "";
		boolean invertmatch = false;
		boolean usechunk = false;
		boolean subtract_baseline = false;
		boolean matching_profile = false;
		boolean logarithmize = false;
		boolean sliding = false;
		boolean counting = false;
		Alphabet.set(new BinaryAlphabet());
		CommandLine cmdline = null;
		try {
			cmdline = parser.parse(myOptions, args);
			n = Integer.parseInt(cmdline.getOptionValue("n"));
			r = Integer.parseInt(cmdline.getOptionValue("r"));
			if (cmdline.hasOption("g"))
				Settings.DEBUG = true;
			if (cmdline.hasOption("v"))
				invertmatch = true;
			if (cmdline.hasOption("k"))
				usechunk = true;
			if (cmdline.hasOption("c"))
				counting = true;
			if (cmdline.hasOption("l"))
				logarithmize = true;
			if (cmdline.hasOption("p")) {
				matching_profile = true;
				r2 = Integer.parseInt(cmdline.getOptionValue("p"));
			}
			// if( cmdline.hasOption("b") ) subtract_baseline = true;
			self = cmdline.getOptionValue("self");
			if (!new File(self).canRead()) {
				throw new IllegalArgumentException("Can't read file " + self);
			}
			String alpha = cmdline.getOptionValue("alphabet");
			if (alpha != null) {
				if (alpha.startsWith("file://")) {
					Alphabet.set(new Alphabet(new File(alpha.substring(7))));
				}
				if (alpha.equals("infer"))
					Alphabet.set(new Alphabet(new File(self)));
				if (alpha.equals("amino"))
					Alphabet.set(new AminoAcidAlphabet());
				if (alpha.equals("binary"))
					Alphabet.set(new BinaryAlphabet());
				if (alpha.equals("binaryletter"))
					Alphabet.set(new BinaryLetterAlphabet());
				if (alpha.equals("damino"))
					Alphabet.set(new DegenerateAminoAcidAlphabet());
				if (alpha.equals("latin"))
					Alphabet.set(new LatinAlphabet());
			} else {
				Alphabet.set(new Alphabet(new File(self)));
			}
			if (cmdline.hasOption("d")) {
				int escape_letters = Integer.parseInt(cmdline.getOptionValue("d"));
				for (int i = 0; i < escape_letters && i < 10; i++) {
					Alphabet.get().letters().add(Character.forDigit(i, 10));
				}
				Collections.sort(Alphabet.get().letters());
			}
			if (r < 0 || n <= 0 || r > n) {
				throw new IllegalArgumentException(
						"Illegal value(s) for n and/or r");
			}
		} catch (Exception e) {
			System.out.print("Error parsing command line: " + e.getMessage()
					+ "\n\n");
			HelpFormatter help = new HelpFormatter();
			help.printHelp("java -jar negsel.jar", myOptions);
			System.exit(1);
		}

		Debug.log("constructing matcher");
		List<PatternTrie> chunks = RChunkPatterns.rChunkPatterns(self, n, r, 0);
		PatternTrie matcher = null;
		ContiguousCountingDAG counter = null;
		long baseline = 0;
		if (!usechunk) {
			if (counting) {
				counter = new ContiguousCountingDAG(chunks, n, r);
				if (subtract_baseline) {
					baseline = counter.countStringsThatMatch(Settings.first_self_string, r - 1);
				}
			} else {
				matcher = RChunkPatterns.rContiguousGraphWithFailureLinks(chunks, n, r);
			}
		}

		// output matching lengths to be used
		int i1 = 0, i2 = 0;
		if (!matching_profile || usechunk) {
			i1 = r;
			i2 = r;
		} else {
			if (r2 > 0 && r2 <= n) {
				i1 = r2;
				i2 = r2;
			} else {
				i1 = 0;
				i2 = n;
			}
		}

		Debug.log("matcher constructed");

		System.out.println("Running with r:" + r + " and n:" + n);

		for (int fileindex = 0; fileindex < testfiles.length; fileindex++) {

			Scanner scan = new Scanner(testfiles[fileindex]);
			Scanner labelscan = new Scanner(labelfiles[fileindex]);

			while (scan.hasNextLine()) {

				String rawline = scan.nextLine().trim();
				Vector<String> linechunks = preprocess(rawline, n, false);
				Vector<Double> scores = new Vector<Double>();

				for (String chunk : linechunks) {
					// String line = scan.nextLine().trim();
					// System.out.println(line + " ");

					int lineindex;
					// if (line.() < n) {
					// System.out.print("NaN");
					// }
					for (lineindex = 0; lineindex <= chunk.length() - n; lineindex++) {
						double[] nmatch = new double[i2 - i1 + 1];
						String l = chunk.substring(lineindex, lineindex + n);
						if (usechunk) {
							// r-chunk detectors
							int i = 0;
							for (PatternTrie chunkmatcher : chunks) {
								if (chunkmatcher.matches(l.substring(i), r) >= r != invertmatch) {
									nmatch[0]++;
								}
								/*
								 * PatternTrie prefixmatcher =
								 * chunkmatcher.destinationNode(l.substring(i,i+r-1));
								 * if( prefixmatcher != null )
								 * nmatch_r_l += prefixmatcher.count(1);
								 * for( PatternTrie postfixmatcher : chunkmatcher.children ){
								 * if( ( postfixmatcher != null &&
								 * postfixmatcher.matches(l.substring(i),r-1) >= r-1 )
								 * != invertmatch ){
								 * nmatch_r_r ++;
								 * }
								 * }
								 */
								i++;
							}
							if (logarithmize) {
								nmatch[0] = Math.log(1 + nmatch[0]) / Math.log(2.);
							}
						} else {
							// r-contiguous detectors
							long last_result = -1;
							for (int i = i1; i <= i2; i++) {
								if (last_result != 0) {
									if (counting) {
										last_result = counter.countStringsThatMatch(l, i)
												- (i < r ? baseline : 0);
									} else {
										int rm = 1;
										while (matcher.matches(l, rm) >= rm && rm <= n) {
											rm++;
										}
										// if( rm > r ){
										// last_result = 1;
										// } else {
										// last_result = 0;
										// }
										last_result = rm - 1;
									}
									// last_result = ;
									if (logarithmize) {
										nmatch[i - i1] += Math.log(1 + last_result) / Math.log(2.);
									} else {
										nmatch[i - i1] += last_result;
									}
								}
							}
						}

						// String output = "";
						// output += line;
						// output += ", ";

						double chunkscore = 0;

						for (double i : nmatch) {
							// System.out.print(line + " ");
							// System.out.print(i + " "); //Number output
							// output += i;
							// output += ", ";
							chunkscore += i;
						}
						scores.add(chunkscore);
					}
				}

				// output += file.toString().substring(path.(), file.toString().());
				// output += ", ";
				// output += Integer.toString(r);
				// output += "\n";

				String output = "";

				// compute different score metrics bases on the lines' chuncks' individual
				// scores
				double avg_score = 0;
				for (double score : scores) {
					avg_score += score;
				}
				avg_score = avg_score / scores.size(); // average score

				double min_score = Collections.min(scores); // minimum score

				double max_score = Collections.max(scores); // maximum score

				double line_score = avg_score; // select one to use in the output

				String label = labelscan.nextLine().trim(); // retrieve the label of the line

				output = String.format("%s, %f, %s, %d, %d", setname[fileindex].toString(), line_score, label, r, n);

				out.println(output);
				// System.out.println(output);
				System.out.flush();
			}

			scan.close();
			labelscan.close();
		}

		// System.out.println("n: "+n);
		// example1();
	}
}
