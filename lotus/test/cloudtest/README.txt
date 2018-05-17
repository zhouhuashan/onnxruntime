Steps to submit a test from your local machine:

1) Download drop app from https://aiinfra.artifacts.visualstudio.com/_apis/drop/client/exe
   Unzip the downloaded file and add lib/net45 dir to your PATH

2. Download cloudtest binary from https://b/queue/cloudtest
   drop get -a -u https://msasg.artifacts.visualstudio.com/DefaultCollection/_apis/drop/drops/CloudTest/a9f054b8d2713d849d040453304400581fa1e910/b5ec9d52-f049-7d1d-08ab-67a19acef86b -d c:\your_local_folder 

3. Add drop and cloudtest to your PATH
   set PATH=%PATH%;C:\tools\drop;C:\tools\cloudtest\retail\amd64\App\Cheetah\Client
   
3. compile lotus

4. Upload a new drop, put onnx_test_runner_vstest.dll into that drop, as well as the config files in this directory

5. submit test job
   CheetahClient.exe -t [BuildRoot]\TestMap.xml -tenant "lotus" -drop "https://aiinfra.artifacts.visualstudio.com/DefaultCollection/_apis/drop/drops/chasun/test/50" -arch "amd64"
   Please change the drop url to yours. But do not change "[BuildRoot]". Type it as it is. It isn't something you need to fill.