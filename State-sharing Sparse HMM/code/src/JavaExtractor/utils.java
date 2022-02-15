package JavaExtractor;
import java.io.File;
import java.io.FileOutputStream;
public class utils {
	public static boolean genreateOutputFiles(String content,String outputPath) {
			
			byte[] sourceByte = content.getBytes();
			if(null != sourceByte)
			{
				try {
					File file = new File(outputPath);		//�ļ�·����·��+�ļ�����
					if (!file.exists()) {	//�ļ��������򴴽��ļ����ȴ���Ŀ¼
						File dir = new File(file.getParent());
						dir.mkdirs();
						file.createNewFile();
					}
					FileOutputStream outStream = new FileOutputStream(file,true);	//�ļ���������ڽ�����д���ļ�
					outStream.write(sourceByte);
					outStream.close();	//�ر��ļ������
					return true;
				} catch (Exception e) {
					e.printStackTrace();
					return false;
				}
			}
			return false;
			
	}
	
}
