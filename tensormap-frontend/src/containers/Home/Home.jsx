import { useNavigate } from "react-router-dom";
import { Upload, FileText, Github } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import * as urls from "../../constants/Urls";
import * as strings from "../../constants/Strings";

function Home() {
  const navigate = useNavigate();

  const handleButtonUpload = () => {
    navigate(urls.DATA_UPLOAD_URL);
  };

  const handleButtonNewProject = () => {
    navigate(urls.DEEP_LEARN_URL);
  };

  const handleButtonGoToGit = () => {
    window.open("https://github.com/scorelab/TensorMap");
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-6">
          <h1 className="text-2xl font-bold">{strings.HOME_MAIN_TITLE}</h1>
          <p className="mt-2 text-muted-foreground">{strings.HOME_MAIN_CONTENT}</p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-6">
          <h2 className="mb-3 text-lg font-semibold">{strings.HOME_ABILITY_TITLE}</h2>
          <ul className="list-inside list-disc space-y-1 text-green-700">
            <li>{strings.HOME_ABILITY_CONTENT_1}</li>
            <li>{strings.HOME_ABILITY_CONTENT_2}</li>
            <li>{strings.HOME_ABILITY_CONTENT_3}</li>
            <li>{strings.HOME_ABILITY_CONTENT_4}</li>
            <li>{strings.HOME_ABILITY_CONTENT_5}</li>
          </ul>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardContent className="flex flex-col items-center p-6 text-center">
            <Upload className="mb-3 h-10 w-10 text-muted-foreground" />
            <h3 className="mb-3 text-lg font-semibold">Upload Data</h3>
            <Button variant="outline" onClick={handleButtonUpload}>
              Upload
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col items-center p-6 text-center">
            <FileText className="mb-3 h-10 w-10 text-muted-foreground" />
            <h3 className="mb-3 text-lg font-semibold">Create new Project</h3>
            <Button variant="outline" onClick={handleButtonNewProject}>
              Create
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col items-center p-6 text-center">
            <Github className="mb-3 h-10 w-10 text-muted-foreground" />
            <h3 className="mb-3 text-lg font-semibold">Contribute at:</h3>
            <Button variant="outline" onClick={handleButtonGoToGit}>
              Contribute
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default Home;
