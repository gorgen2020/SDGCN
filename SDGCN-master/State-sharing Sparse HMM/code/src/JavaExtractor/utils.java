package JavaExtractor;
import java.io.File;
import java.io.FileOutputStream;
public class utils {
	public static boolean genreateOutputFiles(String content,String outputPath) {
			
			byte[] sourceByte = content.getBytes();
			if(null != sourceByte)
			{
				try {
					File file = new File(outputPath);		//文件路径（路径+文件名）
					if (!file.exists()) {	//文件不存在则创建文件，先创建目录
						File dir = new File(file.getParent());
						dir.mkdirs();
						file.createNewFile();
					}
					FileOutputStream outStream = new FileOutputStream(file,true);	//文件输出流用于将数据写入文件
					outStream.write(sourceByte);
					outStream.close();	//关闭文件输出流
					return true;
				} catch (Exception e) {
					e.printStackTrace();
					return false;
				}
			}
			return false;
			
	}
	
}
