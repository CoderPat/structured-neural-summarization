package java2graph;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class ChunkWriter<T> {
    private String pathPrefix;
    private int currentChunkIdx = 0;
    private int maxChunkSize;
    private List<T> unwrittenElements = new ArrayList<>();

    public ChunkWriter(String pathPrefix, int maxChunkSize) {
        this.pathPrefix = pathPrefix;
        this.maxChunkSize = maxChunkSize;
    }

    public void add(T element) {
        unwrittenElements.add(element);
        if (unwrittenElements.size() >= maxChunkSize) {            
            try {
				writeChunk();
			} catch (IOException e) {
				throw new Error("Cannot write to output chunk file: " + e);
			}
        }
    }

    private void writeChunk() throws IOException {
        FileOutputStream output = new FileOutputStream(pathPrefix + '.' + currentChunkIdx + ".json.gz");
        Gson gson = new GsonBuilder().create(); 
        try {
            Writer writer = new OutputStreamWriter(new GZIPOutputStream(output), "UTF-8");
            writer.write(gson.toJson(unwrittenElements));
            writer.close();
        } finally {
            output.close();
        }

        currentChunkIdx++;
        unwrittenElements.clear();
    }

    public void close() {
        try {
            writeChunk();
        } catch (IOException e) {
            throw new Error("Cannot write to output chunk file: " + e);
        }
    }

}