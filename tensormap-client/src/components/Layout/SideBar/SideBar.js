import React, { useState, useEffect } from 'react';
import { Grid, Menu, Segment } from 'semantic-ui-react';
import { withRouter } from 'react-router-dom';
import * as constants from '../../../constants/Strings';
import * as urls from '../../../constants/Urls';

const SideBar = ({ location, history, mainProps }) => {
  const [activeItem, setActiveItem] = useState('');

  useEffect(() => {
    updateActiveItem();
  }, [location.pathname]);

  const updateActiveItem = () => {
    setActiveItem(location.pathname);
  };

  const handleItemClick = (name) => {
    switch (name) {
      case constants.HOME_SIDEBAR:
        history.push(urls.HOME_URL);
        break;
      case constants.DATA_UPLOAD_SIDEBAR:
        history.push(urls.DATA_UPLOAD_URL);
        break;
      case constants.DATA_PROCESS_SIDEBAR:
        history.push(urls.DATA_PROCESS_URL);
        break;
      default:
        history.push(urls.DEEP_LEARN_URL);
        break;
    }
  };

  return (
    <div>
      <Grid style={{ marginTop: '0.5%' }}>
        <Grid.Column width={3}>
          <Menu fluid vertical tabular size="massive">
            <Menu.Item
              name={constants.HOME_SIDEBAR}
              active={activeItem === urls.HOME_URL}
              onClick={() => handleItemClick(constants.HOME_SIDEBAR)}
              style={{ fontWeight: 'bold' }}
            />
            <Menu.Item
              name={constants.DATA_UPLOAD_SIDEBAR}
              active={activeItem === urls.DATA_UPLOAD_URL}
              onClick={() => handleItemClick(constants.DATA_UPLOAD_SIDEBAR)}
              style={{ fontWeight: 'bold' }}
            />
            <Menu.Item
              name={constants.DATA_PROCESS_SIDEBAR}
              active={activeItem === urls.DATA_PROCESS_URL}
              onClick={() => handleItemClick(constants.DATA_PROCESS_SIDEBAR)}
              style={{ fontWeight: 'bold' }}
            />
            <Menu.Item
              name={constants.DEEP_LEARNING_SIDEBAR}
              active={activeItem === urls.DEEP_LEARN_URL}
              onClick={() => handleItemClick(constants.DEEP_LEARNING_SIDEBAR)}
              style={{ fontWeight: 'bold' }}
            />
          </Menu>
        </Grid.Column>

        <Grid.Column stretched width={13}>
          <Segment style={{ minHeight: '90vh', marginRight: '10px', marginBottom: '10px' }}>
            {mainProps.children}
          </Segment>
        </Grid.Column>
      </Grid>
    </div>
  );
};

export default withRouter(SideBar);
